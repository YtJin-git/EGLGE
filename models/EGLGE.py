import torch
import torch.nn as nn
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from models.common import *
import scipy.sparse as sp
from torch.nn.modules.loss import CrossEntropyLoss
from models.gcn import GraphConv, spm_to_tensor
from utils import get_optimizer
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ELassoGCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers):
        super(ELassoGCN, self).__init__()
        # adj = normt_spm(adj, method='in')
        # adj = spm_to_tensor(adj)
        # self.adj = adj.to(device)
        # adj_mask = normt_spm(adj_mask, method='in')
        # adj_mask = spm_to_tensor(adj_mask)
        # self.adj_mask = nn.Parameter(adj_mask).to(device)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x, adj, train_weight=True):
        if train_weight:
            # adj = torch.mul(adj_matrix_mask, self.adj)
            for conv in self.layers:
                x = conv(x, adj)
            return F.normalize(x)
        else:
            return x


class EGLGE(nn.Module):
    def __init__(self, config, dset):
        super(EGLGE, self).__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.gnn_model_name = config.graph_gcn_type

        allattrs = dset.attrs
        allobj = dset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        # self.full_pairs = list(product(dset.attrs, dset.objs))
        self.pairs = dset.pairs
        self.attr_idx = dset.attr2idx
        self.obj_idx = dset.obj2idx
        self.pair_idx = dset.pair2idx

        all_element_words = list(dset.attrs) + list(dset.objs)
        self.attr_obj_displacement = len(dset.attrs)
        self.element_pair_displacement = len(all_element_words)

        self.dict_Obj2IDX = {word: idx for idx, word in enumerate(dset.objs)}
        self.dict_Attr2IDX = {word: idx for idx, word in enumerate(dset.attrs)}

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.additional_visual_params = self.add_visual_tunable_params()
        output_dim = self.clip.visual.output_dim

        self.attr_disentangler = Disentangler(output_dim).cuda()
        self.obj_disentangler = Disentangler(output_dim).cuda()
        if config.graph_init is not None:
            path = config.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings'].to(device)
            self.adj_matrix = graph['adj']
            self.embeddings = embeddings
        else:
            self.adj_matrix = self.adj_from_pairs()

        hidden_layers = self.config.graph_gr_emb
        if config.graph_gcn_type == 'gcn':
            self.gnn = ELassoGCN(config.graph_emb_dim, config.graph_emb_dim, hidden_layers).cuda()

        self.optimizer = get_optimizer(self, config)
        estimator = EstimateAdj(spm_to_tensor(self.adj_matrix), symmetric=True, device=device).to(device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=config.lr_adj)

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width,
                                    bottleneck=self.config.adapter_dim,
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2) 
        for i_block in range(self.clip.visual.transformer.layers):
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj).cuda()
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def update_dict(self, wdict, row,col,data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def adj_from_pairs(self):

        def edges_from_pairs_close_world(pairs):
            weight_dict = {'data':[],'row':[],'col':[]}

            for i in range(self.element_pair_displacement):
                self.update_dict(weight_dict,i,i,1.)

            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.dict_Attr2IDX[attr], self.dict_Obj2IDX[obj] + self.num_attrs

                # a-o, o-a
                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                # c自环
                pair_node_id = idx + self.element_pair_displacement
                self.update_dict(weight_dict,pair_node_id,pair_node_id,1.)

                # c-a, c-o
                self.update_dict(weight_dict, pair_node_id, attr_idx, 1.)
                self.update_dict(weight_dict, pair_node_id, obj_idx, 1.)

                # a-c, o-c
                self.update_dict(weight_dict, attr_idx, pair_node_id, 1.)
                self.update_dict(weight_dict, obj_idx, pair_node_id, 1.)

            return weight_dict

        def edges_from_pairs_open_world(pairs):

            # result
            weight_dict = {'data': [], 'row': [], 'col': []}

            # load the feasible scores
            import os
            for i in range(self.element_pair_displacement):
                self.update_dict(weight_dict, i, i, 1.)

            for pair_idx, (attr, obj) in enumerate(pairs):

                pair_node_id = pair_idx + self.element_pair_displacement
                attr_idx, obj_idx = self.dict_Attr2IDX[attr], self.dict_Obj2IDX[obj] + self.num_attrs

                self.update_dict(weight_dict, attr_idx, obj_idx, 1.)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.)

                # add pair node self-cycle
                self.update_dict(weight_dict, pair_node_id, pair_node_id, 1.)

                #  pair --> element is 1;
                self.update_dict(weight_dict, pair_node_id, attr_idx, 1.)
                self.update_dict(weight_dict, pair_node_id, obj_idx, 1.)

                # element --> pair is feasibility
                self.update_dict(weight_dict, attr_idx, pair_node_id, 1.)
                self.update_dict(weight_dict, obj_idx, pair_node_id, 1.)

            return weight_dict

        if self.config.open_world:
            edges = edges_from_pairs_open_world(self.pairs)
        else:
            edges = edges_from_pairs_close_world(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])),
                            shape=(len(self.pairs) + self.element_pair_displacement, len(self.pairs) + self.element_pair_displacement))

        return adj

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def init_embeddings(self, pairs):
        all_pairs = torch.tensor([(self.attr_idx[attr], self.obj_idx[obj])
                                    for attr, obj in self.pairs]).cuda()
        attr_idx, obj_idx = all_pairs[:, 0], all_pairs[:, 1]
        token_tensor, num_elements = list(), [len(pairs), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element],  1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj).to(device)

        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
            ].type(self.clip.dtype)
        token_tensor[0][
        :, 1: len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)

        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
                                                :self.offset
                                                ].type(self.clip.dtype)
        token_tensor[1][
        :, 1: len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)

        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
                                                self.offset:
                                                ].type(self.clip.dtype)
        token_tensor[2][
        :, 1: len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        init_text_features = list()
        for i_element in range(len(token_tensor)):
            if self.config.open_world and i_element == 0:
                total_elements = len(token_tensor[i_element])
                result = list()
                for start_idx in range(0, total_elements, 256):
                    end_idx = min(start_idx + 256, total_elements)
                    batch = token_tensor[i_element][start_idx:end_idx]
                    _t, _ = self.encode_text(
                        self.token_ids[i_element],
                        batch,
                        enable_pos_emb=self.enable_pos_emb,
                    )
                    result.append(_t.cpu())


                result = torch.cat(result, dim=0)
                idx_text_features = result / result.norm(
                    dim=-1, keepdim=True
                )
                init_text_features.append(idx_text_features)
            elif self.config.dataset == 'cgqa':
                total_elements = len(token_tensor[i_element])
                result = list()
                for start_idx in range(0, total_elements, 50):
                    end_idx = min(start_idx + 50, total_elements)
                    batch = token_tensor[i_element][start_idx:end_idx]
                    _t, _ = self.encode_text(
                        self.token_ids[i_element],
                        batch,
                        enable_pos_emb=self.enable_pos_emb,
                    )
                    batch_features = _t / _t.norm(dim=-1, keepdim=True)
                    result.append(batch_features)
                result = torch.cat(result, dim=0)
                idx_text_features = result / result.norm(
                    dim=-1, keepdim=True
                )
                init_text_features.append(idx_text_features)
            else:
                _text_features, _ = self.encode_text(
                    self.token_ids[i_element],
                    token_tensor[i_element],
                    enable_pos_emb=self.enable_pos_emb,
                )
                idx_text_features = _text_features / _text_features.norm(
                    dim=-1, keepdim=True
                )
                init_text_features.append(idx_text_features)

        # a ,o, c 顺序
        full_embeddings = torch.cat([init_text_features[1], init_text_features[2], init_text_features[0].cuda()], dim=0).cuda()
        return full_embeddings

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        comp_logits, attr_logits, obj_logits, graph_logits, graph_attr_logits, graph_obj_logits = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss_graph = loss_fn(graph_logits, batch_target)
        loss_graph_attr = loss_fn(graph_attr_logits, batch_attr)
        loss_graph_obj = loss_fn(graph_obj_logits, batch_obj)
        loss = loss_comp * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight +\
               loss_graph * self.config.graph_loss_weight + \
               loss_graph_attr * self.config.graph_attr_loss_weight + \
               loss_graph_obj * self.config.graph_obj_loss_weight
        return loss

    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits, graph_logits, graph_attr_logits, graph_obj_logits = predict
        comp_logits = comp_logits + graph_logits * self.config.graph_inference_weight
        attr_logits = attr_logits + graph_attr_logits * self.config.graph_attr_inference_weight
        obj_logits = obj_logits + graph_obj_logits * self.config.graph_obj_inference_weight
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        return comp_logits

    def train_or_not(self, train_flag=True):
        for name, param in self.named_parameters():
            if 'soft_att_obj' in name:
                param.requires_grad = train_flag
            elif 'comp_ctx_vectors' in name:
                param.requires_grad = train_flag
            elif 'attr_ctx_vectors' in name:
                param.requires_grad = train_flag
            elif 'obj_ctx_vectors' in name:
                param.requires_grad = train_flag
            elif 'attr_disentangler' in name:
                param.requires_grad = train_flag
            elif 'obj_disentangler' in name:
                param.requires_grad = train_flag
            elif 'gnn' in name:
                param.requires_grad = train_flag
            elif 'additional_visual_params' in name:
                param.requires_grad = train_flag
        return train_flag

    def forward_for_open(self, batch, text_feats, graph_embeddings):
        batch_img = batch[0].cuda()
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        # c
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[0]
            @ text_feats[self.element_pair_displacement:].t())

        # a
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[1]
            @ text_feats[:self.num_attrs].t())

        # o
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[2]
            @ text_feats[self.num_attrs:self.element_pair_displacement].t())

        # c_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[0]
            @ graph_embeddings[self.element_pair_displacement:].t())

        # a_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[1]
            @ graph_embeddings[:self.num_attrs].t())

        # o_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[2]
            @ graph_embeddings[self.num_attrs:self.element_pair_displacement].t())

        return logits

    def forward(self, batch, pairs):
        batch_img = batch[0].cuda()
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        embeddings = self.init_embeddings(self.pairs)

        # # c
        comp_idx = [self.pair_idx[pair]+self.element_pair_displacement for pair in pairs]
        comp_idx = torch.tensor(comp_idx).to(device)

        norm_adj = self.estimator.normalize()
        current_embeddings = self.gnn(embeddings, norm_adj)

        logits = list()

        # c
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[0]
            @ embeddings[comp_idx].t())

        # a
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[1]
            @ embeddings[:self.num_attrs].t())

        # o
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[2]
            @ embeddings[self.num_attrs:self.element_pair_displacement].t())

        # c_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[0]
            @ current_embeddings[comp_idx].t())

        # a_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[1]
            @ current_embeddings[:self.num_attrs].t())

        # o_graph
        logits.append(
            self.clip.logit_scale.exp()
            * normalized_img_features[2]
            @ current_embeddings[self.num_attrs:self.element_pair_displacement].t())

        return logits


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        self.estimated_adj = nn.Parameter(torch.FloatTensor(adj.shape[0], adj.shape[1]))
        self._init_estimation(adj)
        self.device = device
        self.ori = adj

    def _init_estimation(self, adj):
        with torch.no_grad():
            if isinstance(adj, torch.sparse.Tensor):
                self.estimated_adj.data.copy_(adj.to_dense())
            else:
                self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):  # M, ad
        adj = self.estimated_adj * self.ori.to_dense().to(self.device)
        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)  # +1e-8
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        return mx


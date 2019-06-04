import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from unet import UNetWithResnet50Encoder
from unet import Bridge_128

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, v_dim,
                 reconstruction,
                 layer,
                 size,
                 variant,
                 finetune,
                 use_residual,
                 use_feat_loss,
                 dropout_hid,
                 dropout_unet,
                 logger):

        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reconstruction = reconstruction
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.size = size
        self.layer = layer
        self.finetune = finetune
        self.use_residual = use_residual
        self.use_feat_loss = use_feat_loss
        self.dropout_hid = dropout_hid
        self.dropout_unet = dropout_unet

        if self.reconstruction:
            self.d = nn.Dropout(self.dropout_hid)
            self.G = UNetWithResnet50Encoder(finetune=self.finetune, dropout_unet=self.dropout_unet)
            if size == 128:
                self.B = Bridge_128(self.num_hid, self.num_hid)
            # bridge need to tile to layer4 (512)
            # self.B = bridge(self.num_hid, b_dim)

            logger.write('G learning parameters %.2f M' %
                         (sum(p.numel() for p in self.G.parameters() if p.requires_grad) / 1e6))
            # logger.write('B learning parameters %.2f M' %
            #              (sum(p.numel() for p in self.B.parameters() if p.requires_grad) / 1e6))

    def forward(self, id, p, m, q):
        """Forward

        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        pre_pools = p
        image = m.cuda()
        q = q.cuda()

        if self.finetune: #erase extracted pre_pools activation
            _,pre_pools = self.G.encode(image)

        v_ = pre_pools["layer_%s" % str(self.layer)]
        v_ = v_.view(v_.size(0), v_.size(1), -1).permute(0, 2, 1)
        v_ = v_.cuda()

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_, q_emb)

        v_emb = (att * v_).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        # b = self.B(joint_repr)
        bs = joint_repr.size(0)
        b = joint_repr.view(bs, -1, 1, 1)
        if self.size == 128:
            b = self.B(b)
        g = self.G.decode(self.d(b), pre_pools)

        return logits, g

def build_baseline0_newatt(dataset, num_hid, reconstruction, layer=4, size=64, variant='',
                           finetune=False,
                           use_residual=False,
                           use_feat_loss=False,
                           dropout_hid=False,
                           dropout_unet=False,
                           logger=None):

    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid,
                     dataset.v_dim,
                     reconstruction,
                     layer,
                     size,
                     variant,
                     finetune,
                     use_residual,
                     use_feat_loss,
                     dropout_hid,
                     dropout_unet,
                     logger)

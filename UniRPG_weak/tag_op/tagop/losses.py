
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, scale_label, weight, pred):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        import pdb
        decode_logits = pred[1]
        scale_logits = pred[0]
#         import pdb
#         pdb.set_trace()
        scale_logits_loss = F.cross_entropy(target=scale_label.squeeze(-1), input=scale_logits)
        
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        
        
        bsz = tgt_tokens.size(0)
        loss = 0
        for i in range(bsz):
            loss+=F.cross_entropy(target=tgt_tokens[i,:].unsqueeze(0), input=decode_logits[i,:,:].unsqueeze(0).transpose(1, 2)) * weight[i]
        loss = loss/bsz
        loss+=0.3*scale_logits_loss
    
#         loss = F.cross_entropy(target=tgt_tokens, input=decode_logits.transpose(1, 2))
#         loss+=0.3*scale_logits_loss
        return loss


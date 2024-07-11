import torch
from transformers import AutoModelForMaskedLM
import os

class SPLADE(torch.nn.Module):
    
    @staticmethod
    def splade_max(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        output = output.logits
        relu = torch.nn.ReLU(inplace=False)
        values, _ = torch.max(torch.log(1 + relu(output)) * attention_mask.unsqueeze(-1), dim=1)
        return values

    @staticmethod
    def passthrough(output, attention_mask):
        # tokens: output of a huggingface tokenizer
        return output

    def train(self, mode=True):
        if self.freeze_d_model: 
            self.query_encoder.train(mode)
            self.doc_encoder.train(False)
        else: 
            self.query_encoder.train(mode)
            self.doc_encoder.train(mode)


    def __init__(self, model_type_or_dir, tokenizer=None, shared_weights=True, n_negatives=-1, splade_doc=False, model_q=None, 
                 freeze_d_model=False,
                 #adapter_name: str = None,
                 #adapter_config: str = None, #,Union[str, AdapterConfig] = None, 
                 #load_adapter: str = None,
                 **kwargs):
        """
        output indicates which representation(s) to output ('MLM' for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        self._keys_to_ignore_on_save = None
        self._keys_to_ignore_on_load_missing = None
        
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        
        self.output_dim=self.doc_encoder.config.vocab_size

        self.n_negatives = n_negatives
        self.splade_doc = splade_doc
        self.doc_activation = self.splade_max
        self.query_activation = self.splade_max if not self.splade_doc else self.passthrough
        self.freeze_d_model = freeze_d_model

        if shared_weights:
            self.query_encoder = self.doc_encoder
        else:
            if model_q:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_q)
            else:
                self.query_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

        if freeze_d_model:
            assert model_q and not shared_weights
            self.doc_encoder.requires_grad_(False)
            self.doc_encoder.train(False)
        
    def forward(self, **tokens):

        if not self.shared_weights or self.splade_doc:
            attention_mask = tokens["attention_mask"]
            input_ids = tokens["input_ids"] ##(bsz * (nb_neg+2) , seq_length)
            input_ids = input_ids.view(-1,self.n_negatives+2,input_ids.size(1)) ##(bsz, nb_neg+2 , seq_length)
            attention_mask = attention_mask.view(-1,self.n_negatives+2,attention_mask.size(1))
            docs_ids = input_ids[:,1:,:].reshape(-1,input_ids.size(2)) ##(bsz * (nb_neg+1) , seq_length)
            docs_attention = attention_mask[:,1:,:].reshape(-1,attention_mask.size(2))
            queries_ids = input_ids[:,:1,:].reshape(-1,input_ids.size(2))  ##(bsz * (1) , seq_length)
            queries_attention = attention_mask[:,:1,:].reshape(-1,attention_mask.size(2))

            queries_result = self.query_activation(self.query_encoder(input_ids=queries_ids,attention_mask=queries_attention), attention_mask=queries_attention)
            queries_result = queries_result.view(-1,1,queries_result.size(1))  ##(bsz, (1) , Vocab)
            docs_result = self.doc_activation(self.doc_encoder(input_ids=docs_ids,attention_mask=docs_attention),attention_mask=docs_attention)
            docs_result = docs_result.view(-1,self.n_negatives+1,docs_result.size(1))  ####(bsz, (nb_neg+1) , Vocab)
        else:
            representations = self.doc_activation(self.doc_encoder(**tokens),attention_mask=tokens["attention_mask"]) #TODO This should separate docs and queries and use their separate activations, for now is not a problem because they will always be the same if we are here.
            output = representations.view(-1,self.n_negatives+2,representations.size(1))
            queries_result = output[:,:1,:]
            docs_result = output[:,1:,:]
        return queries_result,docs_result

    def save(self,output_dir, tokenizer):
        model_dict = self.doc_encoder.state_dict()
        torch.save(model_dict, os.path.join(output_dir,  "pytorch_model.bin"))
        self.doc_encoder.config.save_pretrained(output_dir)

        if not self.shared_weights:
            query_output_dir = os.path.join(output_dir,"query")
            os.makedirs(query_output_dir, exist_ok=True)
            self.query_encoder.save_pretrained(query_output_dir)
            self.query_encoder.config.save_pretrained(query_output_dir)
            if tokenizer:
                tokenizer.save_pretrained(query_output_dir)

        if tokenizer:
            tokenizer.save_pretrained(output_dir)

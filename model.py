import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaForQuestionAnswering, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class CustomModel(RobertaForQuestionAnswering): 
    def __init__(self, config):
        super().__init__(config) 
        
        # cnn
        self.conv1d_layer = nn.Conv1d(config.hidden_size, 1024, kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        self.dense_layer = nn.Linear(1024, 2, bias=True)

        # lstm
        # self.lstm = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.5, bidirectional = True, batch_first = True)
        # self.dense_layer = nn.Linear(2048, config.num_labels, bias=True)

        self.init_weights()
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        head_mask=None, 
        inputs_embeds=None, 
        start_positions=None, 
        end_positions=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # cnn
        sequence_output = outputs[0]
        conv_input = sequence_output.transpose(1, 2)
        conv_output = F.relu(self.conv1d_layer(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output = conv_output.transpose(1, 2)
        conv_output = self.dropout(conv_output)
        logits = self.dense_layer(conv_output)

        # lstm
        # sequence_output = outputs[0]
        # lstm_output, (last_hidden, last_cell) = self.lstm(sequence_output)
        # logits = self.dense_layer(lstm_output)
        
        # 공통
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
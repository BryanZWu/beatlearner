import torch

class ConditionedSparseAttention(torch.nn.Module):
    def __init__(self, num_heads, num_features, attention_window, in_channels):
        '''
        Args:
            num_heads: Number of attention heads
            num_features: Number of features in the input
            attention_window: Number of frames to attend to. This applies to both the 
                attention to the condition and the attention to the input
            in_channels: Number of channels in the input
        '''
        super(ConditionedSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        self.attention_window = attention_window
        self.in_channels = in_channels


    def forward(self, x, condition, condition_mask, x_mask):
        '''
        Args:
            x: (Batch, Time, Channels). The Map
            condition: (Batch, Time, Channels). The Condition, in our case, the audio encoding
        '''
        # Attention that attents to both the last self.attention_window frames of x
        # and the last self.attention_window frames of condition
        # optional: attend to future condition? 
        
        # an efficient version of
        # for i in range(attn_window, Time):
        #    attend_to_things = torch.cat([x[:, i-attn_window:i], condition[:, i-attn_window:i+attn_window(?)]], dim=1)
        #    attended = attention(attend_to_things) # Find attention impl online

        # Make sure that the none of these depend on fixed length (workign w seq)






# class MapNet(torch.nn.Module):
#     def __init__(self):
#         super(MapNet, self).__init__()
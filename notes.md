The attention_weights key contains the self-attention weights learnt by the transformer layers. I would suggest watching the rasa algorithm whiteboard series on transform to understand more about self-attention and what this relates to - Rasa Algorithm Whiteboard - Transformers & Attention 1: Self Attention 

text_transformed is the output coming out of the transformer layer. It’s the sequence of vectors created for each input token.

The value of the __CLS__ token can be retrieved by taking the last token’s vector from text_transformed . This is the vector that is passed to the Embedding Layer after the transformer in the figure.



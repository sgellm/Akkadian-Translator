Akkadian Translator

The goal of this project is to create a translator between Akkadian cuneiform and English. Since Akkadian has not been written or spoken since roughly 100 AD,
this project is largely reliant on HuggingFace datasets for the text. I hope that this project can be beneficial to those who want to learn more about languages
or Mesopotamian culture as a whole.

Architecture:

As a tokenizer, I chose to create a single tokenizer using the ByteLevelBPETokenizer framework from the tokenizers library. Since Akkadian is reliant on symbols
with no spaces denoting the end of a sentence, I chose ByteLevelBPE because I think that it would be better for such an under resourced language that does not
have clear delimiters. The model was also trained with separate tokenizers for English and Akkadian, which performed worse. For the actual model, I chose an 
encoder-decoder architecture to implement the Seq2Seq model. I used PyTorch tensors to represent the sequences. For a project with such fragmented data, I begin
the model with a high learning rate (0.01), but slowly decrease it throughout the project. I did the same with the teacher forcing ratio as well, slowly decreasing
it from a high amount. I hope that this allows the model to gain a lot of knowledge early on, and progress to make mistakes and fix itself later on after it has
gone through a lot of iterations.

What next:

I plan to continue testing this model and improving it. There are a variety of hyperparameters that can be changed to potentially improve model performance, such
as learning rate, hidden size, or teacher forcing ratio. Furthermore, I also plan to try to test a Seq2Seq model between English and Akkadian transliterated text.

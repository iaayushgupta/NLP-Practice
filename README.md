# NLP-Practice

There is a library called as huggingface that you can pip install, import                

    This library has trained models that we can directly use for text vectorization in NLP tasks            

    It has BERT-base-uncased, BERT-base-case, BERT-large-uncased etc            

        base model means smaller BERT model (12 layers, large means 24 layers) and cased/uncased means allowing large, small chars or not        

        It also has BERT models based on several languages        

        It also has pretrained GPT models from OpenAI, Roberta models        

    To install huggingface library.. just write pip install transformers            

        the library itself is called transformers        

Now, there is a concept called as sub-tokenization. Lets see what it is:                

    We know tokenization means, we split sentence/text into words and assign an index (scalar) to each word            

    This is called word tokenization            

    We also learnt about character tokenization earlier            

    Now, sub-tokenization comes mid-way between word tokenization and char tokenization            

    Take an example of 2 words: fast, faster            

        In word tokenization, these 2 will get 2 completely different indexes (tokens)        

        In char tokenization, each char gets different index        

        In sub-tokenization, faster will be split into fast + er        

        So, the word token faster, instead of being split into 6 char, like in char tokenization, will now be split into just 2 words i.e. fast and er        

            fast will have one index and er will be given another index    

    This is the concept of sub-tokenization            

Different libraries/codes use different types of tokenization. Some models in huggingface have used sub-tokenization, not all models in huggingface has used it though                

Remember, how to tokenize is as important as deciding, say which model to use etc                

huggingface provides you with a class, that you can use, called BERTTokenizer                

    It you give it some text, it will tokenize it based on the way needed by BERT            

        BERT needs tokenization to be done in a special form        

    There is also a class called BertTokenizerFast, which is used for tokenizing large texts            

    These tokenizers are pretrained so that you get some output based on input word            

    Code:            

        from transformers import BertTokenizer        

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')        

            So this form of tokenizing is tuned for bert-base-cased model    

Now, remember that for any text input to BERT, we always start with special word called as [CLS] which indicates start of the text/seq                

Also, [SEP] implies fullstop or end of sentence                

Another special word is [MASK], which we were using while training BERT model (to mask and predict words in input seq/text)                

    print(bert_tokenizer.cls_token)            

        returns [CLS]        

    Now, lets encode a text and see the value of all tokens in that text            

    enc = bert_tokenizer.encode("Hi, I am James Bond !")            

    print(enc)            

        gives [101, 8790, 117, 146, 1821, 1600, 7069, 106, 102]        

    print(bert_tokenizer.decode(enc))            

        gives [CLS] Hi, I am James bond! [SEP]        

    print(bert_tokenizer.decode([117]))            

        gives ,        

    print(bert_tokenizer.decode([106]))            

        gives !        

There have been many other models for NLP tasks after BERT (published in 2018) e.g. GPT, RoBERTa, DistilBERT to name a few, having several millions of trainable parameters                

    To know which model to use, you need to check what is the hardware reqmnt by the model            

        Some of these models need cutting edge GPUs which might be available only with big corporates        

    Some of these models were trained on large corpus of data needing a lot of time to train too e.g. say 5 days in some models v/s 1 day in some other models            

DistilBERThas the fewest number of parameters of all above models (but training time is much more for this model). Performance is slightly lower than other models, but neglegible                

DistilBERT is the most popular choice for transfer learning nowadays since even during evaluation, it is very fast and performance is very close to BERT                

If you still want state-of-the-art model, you should use RoBERTa model or XLNet (which is another model)                 

    You can use any of these models with huggingface library with just 1-line code change            

                

Now, lets see how to use DistilBERT                

    There are primarily 2 types of models in DistilBERT: ones for using in Pytorch and others for Tensorflow            

    Models to be used in Tensorflow have their class names starting with TF            

Code for text featurization:                

    import tensorflow as tf            

    from transformers import DistilBertTokenizer, TFDistilBertModel            

    # Loading pretrained existing tokenizer which has been best tuned for base-uncased model            

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')            

    # Loading pretrained model for base-uncased data (trained parameters)            

    model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')            

    # For text featurization, we learnt that the 768 dimensional o/p we get for [CLS] can be considered as the feature vector for the whole string/text            

    # Tokenizing text for inferencing            

    e = tokenizer.encode("Hello, my dog is cute")            

    # Building a model            

    input = tf.constant(e)[None, :]            

    # Input to a model is in batches.             

    # Since e is a list, we have converted it into a tf.constant            

    # Thus, input shape = (None, # tokens in each batchitem). This ends up being (1,#tokens in the text)            

    output = model(input)            

    print(len(output)) # gives 1            

        The output is a tuple of size 1 i.e. only 1 element in the tuple        

        0th element of this tuple i.e. output[0] : is tensor of shape (1,#tokens,768)  i.e. 768 dimensional o/p for each token        

        We only need token corresponding to [CLS] i.e. 0th token from above tensor        

        But, if you remember, we discussed earlier in BERT that, instead of taking just o/p of the last layer of the model, it is better to take o/p from last 4 hidden layers        

        In order to do this, we use another class provided by huggingface called DistilBERTConfig        

Code for o/p from last 4 hidden layers:                

    from transformers import  DistilBertConfig            

    config = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True)            

    # When we set the config with hidden_states = True, we are telling that we want model output at all layers            

    e = tokenizer.encode("Hello, my dog is cute")            

    input = tf.constant(e)[None, :]            

    model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config = config)            

    print(model.config)            

    # Every model has config as one parameter. We have set hidden_states=True in this config file            

    output = model(input)            

    # Now that model is built, lets check size of o/p layer            

    print(len(output)) # Gives 2 instead of 1 when hidden_states in config was False (by default it is False)            

    # Now lets check what are the 2 components in this output tuple            

    # output[0] gives the same shape as when hidden_states was False.. i.e. it gives only the last layer output. The shape is  (1,#tokens,768) just like earlier            

    # output[1] means other layers.. in this we can access ith layer output using output[1][i]            

    # But len(output[1]) = 7 i.e. it is a 6 layer model.. (not sure why 7th layer)            

    # Shape of output[1][i] will be  (1,#tokens,768)            

    # Now we can simply concatenate o/ps of the last 4 layers for [CLS] and use it for as featurization for the text input            

The way that BERT is trained usually is using 2 types of datasets:                

    1) Masked Language Modelling: Here some tokens are masked and task is to predict the masked token            

    2) Next Sentence Prediction: Here 2 sentences are given and task is to predict if last sentence is next sentence or not (classification task) 

import os
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import random
import pandas as pd 
import numpy as np 
import re 
import gensim
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset, DataLoader


DEVICE = "cuda:0"
punctuation = '"$%&\'()*+-/:<=>[\\]^_`{|}~•@;#…!'    # define a string of punctuation symbols
punctuation += "'"

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet

def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = tweet.lower()  # lower case
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    # tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    # tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    # tweet = ' '.join(tweet_token_list)
    return tweet


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.full_text[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True, 
            padding="max_length"
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def train(epochs, model, training_loader, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for epoch in range(epochs):
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.long)

            model.zero_grad() 
            outputs = model(ids, attention_mask=mask, labels=targets)
            # loss = loss_function(outputs.logits, targets)
            loss = outputs.loss
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.logits, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")

    model.save_pretrained("./outputs/models/covidhera_onlytext_V2_clim4.pt")


def valid(model, testing_loader):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.long)

            outputs = model(ids, attention_mask=mask, labels=targets)
            loss = outputs.loss
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.logits, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu


def main():
    seed_everything(0)
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 3
    LEARNING_RATE = 1e-05

    #load ctbert model
    # model_name = "digitalepidemiologylab/covid-twitter-bert-v2" 
    model_name = "./outputs/models/covidhera_onlytext_V2_acc_9888.pt"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    #model = AutoModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")


    #load data
    data = pd.read_csv("./outputs/covidhera/input.csv") # index,id,full_text,title,label


    #preprocess text
    #counts : 87742 info | 8128 misinfo
    balanced_data = data[data["label"] == 1]
    info_data = data[data["label"] == 0]

    info_data = info_data.loc[np.random.choice(info_data.index, 8128, replace=False)]
    balanced_data = pd.concat([balanced_data, info_data])
    
    balanced_data['full_text'] = balanced_data.full_text.apply(preprocess_tweet)
    balanced_data['full_text'] = balanced_data.full_text.apply(remove_emoji)
    balanced_data = balanced_data[balanced_data['full_text'].str.len() > 3]

    balanced_data['title'] = balanced_data.title.apply(preprocess_tweet)
    balanced_data['title'] = balanced_data.title.apply(remove_emoji)
    balanced_data = balanced_data[balanced_data['title'].str.len() > 3]


    ###################### only on full_text ############################
    # Creating the dataset and dataloader for the neural network
    balanced_data = balanced_data[["full_text", "label"]]
    train_size = 0.8
    train_dataset = balanced_data.sample(frac=train_size, random_state=42)
    test_dataset = balanced_data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    balanced_data.to_csv("./outputs/test.csv", index=False)

    print("FULL Dataset: {}".format(balanced_data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Triage(train_dataset, tokenizer, MAX_LEN)
    testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    train(EPOCHS, model, training_loader, loss_function, optimizer)

    #validation
    acc = valid(model, testing_loader)
    print("Accuracy on test data = %0.2f%%" % acc)

    



if __name__ == "__main__":
    main()

































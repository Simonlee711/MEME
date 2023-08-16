from torch.utils.data import DataLoader

def dataloader(tokenized_dataset):
    '''
    Write a dataloader
    '''

    train_dataloader = DataLoader(
        tokenized_dataset['train'], shuffle = True, batch_size = 32, collate_fn = data_collator
    )

    eval_dataloader = DataLoader(
        tokenized_dataset['valid'], shuffle = True, collate_fn = data_collator
    )
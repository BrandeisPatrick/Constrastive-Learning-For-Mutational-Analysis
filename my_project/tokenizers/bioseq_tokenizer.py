# my_project/tokenizers/bioseq_tokenizer.py

import torch
from typing import List, Dict, Optional
from collections import defaultdict


class BioSeqTokenizer:
    """
    Custom tokenizer for biological sequences.

    Args:
        alphabets (Optional[List[str]], optional): List of alphabet characters. Defaults to None.
        window (int, optional): Window size for n-grams. Defaults to 1.
        stride (int, optional): Stride for n-gram generation. Defaults to 1.
        model (str, optional): Model type. Defaults to 'Encoder'.
    """

    def __init__(self, alphabets: Optional[List[str]] = None, window: int = 1,
                 stride: int = 1, model: str = 'Encoder'):
        if alphabets is None:
            alphabets = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        specials = ['<PAD>', '<CLS>', '<SOS>', '<EOS>', '<SEP>', '<UNK>', '<MASK>']

        if stride < 1 or window < 1:
            raise ValueError('Invalid parameters for window or stride. Must be >= 1.')

        self.model = model
        self.alphabets = alphabets
        self.window = window
        self.stride = stride
        self.specials = specials

        self.words = self.specials + self.generate_vocab(self.alphabets, self.window)
        self.tokens = torch.arange(len(self.words)).tolist()
        self.word2token: Dict[str, int] = dict(zip(self.words, self.tokens))
        self.token2word: Dict[int, str] = dict(zip(self.tokens, self.words))
        self.ntoken = len(self.words)
        self.special_tokens = [self.word2token[x] for x in self.specials]

        # Special token IDs
        self.sos_token = self.word2token['<SOS>']
        self.eos_token = self.word2token['<EOS>']
        self.cls_token = self.word2token['<CLS>']
        self.mask_token = self.word2token['<MASK>']
        self.pad_token_id = self.word2token['<PAD>']

    def generate_vocab(self, alphabets: List[str], n: int = 1) -> List[str]:
        """
        Generates a vocabulary based on the provided alphabets and n-gram size.

        Args:
            alphabets (List[str]): List of characters.
            n (int, optional): N-gram size. Defaults to 1.

        Returns:
            List[str]: Generated vocabulary.
        """
        if n == 1:
            return alphabets
        else:
            vocab = alphabets.copy()
            for _ in range(n - 1):
                vocab = [x + y for x in vocab for y in alphabets]
            return vocab

    def tokenize(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Tokenizes a list of sequences into token IDs.

        Args:
            sequences (List[str]): List of biological sequences.

        Returns:
            List[torch.Tensor]: List of token ID tensors.
        """
        tokens_list = []
        for seq in sequences:
            tokens = self.preprocess(seq)
            tokens = ['<CLS>'] + tokens  # Add start token
            token_ids = [self.word2token.get(word, self.word2token['<UNK>']) for word in tokens]
            tokens_list.append(torch.tensor(token_ids, dtype=torch.long))
        return tokens_list

    def preprocess(self, sequence: str) -> List[str]:
        """
        Preprocesses a sequence into n-grams.

        Args:
            sequence (str): Biological sequence.

        Returns:
            List[str]: List of n-grams.
        """
        sequence = sequence.upper()
        n = self.window
        return [sequence[i:i + n] for i in range(0, len(sequence) - n + 1, self.stride)]

    def textify(self, token_ids: List[int], remove_special_tokens: bool = False) -> str:
        """
        Converts token IDs back to a biological sequence.

        Args:
            token_ids (List[int]): List of token IDs.
            remove_special_tokens (bool, optional): Flag to remove special tokens. Defaults to False.

        Returns:
            str: Reconstructed biological sequence.
        """
        words = [
            self.token2word[token] if token not in self.special_tokens or not remove_special_tokens
            else 'X' for token in token_ids
        ]
        return ''.join(words)

    def convert_tokens_to_ids(self, tokens: List[str]) -> torch.Tensor:
        """
        Converts a list of tokens to their corresponding IDs.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            torch.Tensor: Tensor of token IDs.
        """
        token_ids = [self.word2token.get(token, self.word2token['<UNK>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.ntoken
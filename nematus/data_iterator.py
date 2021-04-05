import numpy

import gzip

import shuffle
from util import load_dict, reverse_dict
import random

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding="UTF-8")
    return open(filename, mode, encoding="UTF-8")

class FileWrapper(object):
    def __init__(self, fname):
        self.pos = 0
        self.lines = fopen(fname).readlines()
        self.lines = numpy.array(self.lines, dtype=numpy.object)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos >= len(self.lines):
            raise StopIteration
        l = self.lines[self.pos]
        self.pos += 1
        return l
    def reset(self):
        self.pos = 0
    def seek(self, pos):
        assert pos == 0
        self.pos = 0
    def readline(self):
        return next(self)
    def shuffle_lines(self, perm):
        self.lines = self.lines[perm]
        self.pos = 0
    def __len__(self):
        return len(self.lines)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, mt, target,
                 source_dicts, target_dict,
                 model_type,
                 batch_size=128,
                 maxlen=100,
                 source_vocab_sizes=None,
                 target_vocab_size=None,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 token_batch_size=0,
                 keep_data_in_memory=False,
                 data_mode='single',
                 is_train=False,
                 beam_mode_pro=0.0,
                 noise_mode_pro=0.0,
                 sample_mode_pro=1.0,
                 beam_dropout=0.0,
                 noise_dropout=0.0,
                 sample_dropout=0.0,
                 chunk_size=10000000,
                 noise_source=False,
                 f_ratio=0.0,
                 f_source=None,
                 f_mt=None,
                 f_target=None):
        if keep_data_in_memory:
            if f_ratio != 1:
                self.source, self.mt, self.target = FileWrapper(source), FileWrapper(mt), FileWrapper(target)
            if f_ratio:
                self.f_source, self.f_mt, self.f_target = FileWrapper(f_source), FileWrapper(f_mt), FileWrapper(f_target)
            if shuffle_each_epoch:
                if f_ratio != 1:
                    r = numpy.random.permutation(len(self.source))
                    self.source.shuffle_lines(r)
                    self.mt.shuffle_lines(r)
                    self.target.shuffle_lines(r)
                if f_ratio:
                    r = numpy.random.permutation(len(self.f_source))
                    self.f_source.shuffle_lines(r)
                    self.f_mt.shuffle_lines(r)
                    self.f_target.shuffle_lines(r)
        elif shuffle_each_epoch:
            self.chunk_size = chunk_size
            if f_ratio != 1:
                self.source_orig = source
                self.mt_orig = mt
                self.target_orig = target
                # three file objects, storing shuffled temporarily parallel datasets
                self.source, self.mt, self.target = shuffle.jointly_shuffle_files(
                    [self.source_orig, self.mt_orig, self.target_orig], chunk_size=self.chunk_size, temporary=True)
            if f_ratio:
                self.f_source_orig = f_source
                self.f_mt_orig = f_mt
                self.f_target_orig = f_target
                # three file objects, storing shuffled temporarily parallel datasets
                self.f_source, self.f_mt, self.f_target = shuffle.jointly_shuffle_files(
                    [self.f_source_orig, self.f_mt_orig, self.f_target_orig], chunk_size=self.chunk_size, temporary=True)
        else:
            if f_ratio != 1:
                self.source = fopen(source, 'r')
                self.mt = fopen(mt, 'r')
                self.target = fopen(target, 'r')
            if f_ratio:
                self.f_source = fopen(f_source, 'r')
                self.f_mt = fopen(f_mt, 'r')
                self.f_target = fopen(f_target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict, model_type))
        self.target_dict = load_dict(target_dict, model_type)
        if data_mode == 'multiple' or data_mode == 'ape' :
            self.re_target_dict = reverse_dict(self.target_dict)
            self.re_source_dicts = [reverse_dict(d) for d in self.source_dicts]
        self.noise_source = noise_source

        # Determine the UNK value for each dictionary (the value depends on
        # which version of build_dictionary.py was used).

        def determine_unk_val(d):
            if '<UNK>' in d and d['<UNK>'] == 2:
                return 2
            return 1

        self.source_unk_vals = [determine_unk_val(d)
                                for d in self.source_dicts]
        self.target_unk_val = determine_unk_val(self.target_dict)


        self.keep_data_in_memory = keep_data_in_memory
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.data_mode = data_mode
        self.is_train = is_train
        self.beam_mode_pro = beam_mode_pro
        self.noise_mode_pro = noise_mode_pro
        self.sample_mode_pro = sample_mode_pro
        if is_train:
            self.beam_dropout = beam_dropout
            self.noise_dropout = noise_dropout
            self.sample_dropout = sample_dropout
        else:
            self.beam_dropout = 0.0
            self.noise_dropout = 0.0
            self.sample_dropout = 0.0

        self.f_ratio = f_ratio
        self.fine_tune = False


        if self.data_mode == 'multiple' or self.data_mode == 'ape':
            self.mode_probs = {'sample': self.sample_mode_pro, 'noise': self.noise_mode_pro, 'beam': self.beam_mode_pro} \
                if is_train else {'sample': 0, 'noise': 0, 'beam': 1}

        self.source_vocab_sizes = source_vocab_sizes
        self.target_vocab_size = target_vocab_size

        self.token_batch_size = token_batch_size

        if self.source_vocab_sizes != None:
            assert len(self.source_vocab_sizes) == len(self.source_dicts)
            for d, vocab_size in zip(self.source_dicts, self.source_vocab_sizes):
                if vocab_size != None and vocab_size > 0:
                    for key, idx in list(d.items()):
                        if idx >= vocab_size:
                            del d[key]

        if self.target_vocab_size != None and self.target_vocab_size > 0:
            for key, idx in list(self.target_dict.items()):
                if idx >= self.target_vocab_size:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.mt_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            if self.keep_data_in_memory:
                if self.fine_tune:
                    r = numpy.random.permutation(len(self.f_source))
                    self.f_source.shuffle_lines(r)
                    self.f_mt.shuffle_lines(r)
                    self.f_target.shuffle_lines(r)
                else:
                    r = numpy.random.permutation(len(self.source))
                    self.source.shuffle_lines(r)
                    self.mt.shuffle_lines(r)
                    self.target.shuffle_lines(r)
            else:
                if self.fine_tune:
                    self.f_source, self.f_mt, self.f_target = shuffle.jointly_shuffle_files(
                        [self.f_source_orig, self.f_mt_orig, self.f_target_orig], chunk_size=self.chunk_size,
                        temporary=True)
                else:
                    self.source, self.mt, self.target = shuffle.jointly_shuffle_files(
                        [self.source_orig, self.mt_orig, self.target_orig], chunk_size=self.chunk_size, temporary=True)
        else:
            if self.fine_tune:
                self.f_source.seek(0)
                self.f_mt.seek(0)
                self.f_target.seek(0)
            else:
                self.source.seek(0)
                self.mt.seek(0)
                self.target.seek(0)

    def read_sample(self, sample):
        probs = []
        sents = []
        for el in sample.strip().split(" _eos "):
            probs.append(float(el.split(':')[-1]))
            sents.append(':'.join(el.split(':')[:-1]).strip())
        num_samples = sum(probs)
        probs_ = [p / num_samples for p in probs]
        return sents, probs_

    def word_dropout(self, sent, dropout, dic, tag = False):
        """sent is a string of sentence, without _eos_ token"""
        sent = sent.split()
        drop_mask = (numpy.random.random(len(sent)) < dropout).astype(int)

        if tag == True:
            # deleting with probability of beam_drop
            for i in range(len(drop_mask)-1, -1, -1):
                if drop_mask[i] == 1:
                    sent.pop(i)
            # replacing
            drop_mask = (numpy.random.random(len(sent)) < dropout).astype(int)
            # TODO: make sure all special tokens are ranked on the top of dictionary and included in low=max()
            replacement = numpy.random.randint(low=max(dic['<EOS>'], dic['<GO>'], dic['<UNK>']) + 1, high=len(dic) - 1,
                                               size=len(sent))
            for i, item in enumerate(drop_mask):
                if item == 1:
                    sent[i] = self.re_source_dicts[0][replacement[i]]
            # swapping
            drop_mask = (numpy.random.random(len(sent)) < dropout).astype(int)
            for i, item in enumerate(drop_mask):
                if item == 1:
                    index = random.choice([-3,-2,-1,1,2,3])
                    if index > 0:
                        if i+index >= len(sent):
                            index = -index
                        if i+index < 0:
                            continue
                    else:
                        if i+index < 0:
                            index = -index
                        if i+index >= len(sent):
                            continue
                    sent[i], sent[i+index] = sent[i+index], sent[i]
        else:
            # TODO: make sure all special tokens are ranked on the top of dictionary and included in low=max()
            replacement = numpy.random.randint(low=max(dic['<EOS>'], dic['<GO>'], dic['<UNK>']) + 1, high=len(dic) - 1,
                                               size=len(sent))
            for i, item in enumerate(drop_mask):
                if item == 1:
                    sent[i] = self.re_target_dict[replacement[i]]
        return sent

    def get_base_translation(self, ind, beam_sents, out_sents, samples, tag=False):
        if tag == True:
            mode = 'beam'
        else:
            keys, probs = map(list, zip(*self.mode_probs.items()))
            mode = numpy.random.choice(keys, p=probs)
        if mode == 'sample':
            dropout = self.sample_dropout
            sents_, probs_ = self.read_sample(samples[ind])
            base_translation = numpy.random.choice(sents_, p=probs_)
        elif mode == 'noise':
            dropout = self.noise_dropout
            base_translation = out_sents[ind]
        elif mode == 'beam':
            dropout = self.beam_dropout
            base_translation = beam_sents[ind]

        if tag == True:
            base_translation = self.word_dropout(base_translation,
                                                 dropout=dropout,
                                                 dic=self.source_dicts[0], tag=True)
        else:
            base_translation = self.word_dropout(base_translation,
                                            dropout=dropout,
                                            dic=self.target_dict)

        return base_translation

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        mt = []
        target = []

        longest_source = 0
        longest_mt = 0
        longest_target = 0

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.mt_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            while True:
                if random.random() < self.f_ratio:
                    s_file, m_file, t_file = self.f_source, self.f_mt, self.f_target
                    # not allow source-side noisy for fine-tuning data, but keep mt-side noisy consistent with sythetic data
                    self.fine_tune = True
                else:
                    s_file, m_file, t_file = self.source, self.mt, self.target
                    self.fine_tune = False
                ss = s_file.readline()
                tt = t_file.readline()
                mm = m_file.readline()
                # check whether file finished reading
                if not ss:
                    break
                if self.data_mode == 'multiple':
                    # generate mt-side sentence
                    beam_sents = [i.strip() for i in mm.split(" _eos_eos ")[0].split(" _eos ")]
                    out_sents = [i.strip() for i in tt.split(" _eos ")]
                    samples = mm.split(" _eos_eos ")[1:]
                    if self.is_train:
                        assert len(samples) == len(beam_sents) == len(out_sents)

                    current_inp = []
                    for i in range(len(beam_sents)):
                        base_translation = self.get_base_translation(i, beam_sents, out_sents, samples)
                        current_inp = current_inp + base_translation + ["_eos"]
                    mm = current_inp[:-1]

                    if self.noise_source == True and self.fine_tune == False:
                        # generate noisy source-side data
                        beam_sents = [i.strip() for i in ss.split(" _eos ")]
                        samples, out_sents = None, None
                        current_inp = []
                        for i in range(len(beam_sents)):
                            base_translation = self.get_base_translation(i, beam_sents, out_sents, samples, True)
                            current_inp = current_inp + base_translation + ["_eos"]
                        ss = current_inp[:-1]
                elif self.data_mode == 'ape' and self.is_train:
                    samples = [mm]
                    beam_sents, out_sents = None, None
                    mm = self.get_base_translation(0, beam_sents, out_sents, samples)
                else:
                    mm = mm.split()

                if self.noise_source == False or self.fine_tune == True:
                    ss = ss.split()
                tt = tt.split()

                if self.skip_empty and (len(ss) == 0 or len(tt) == 0 or len(mm) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen or len(mm) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.mt_buffer.append(mm)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.mt_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by source/target buffer length
            if self.sort_by_length:
                tlen = numpy.array([max(len(s),len(m), len(t)) for (s,m,t) in zip(self.source_buffer,self.mt_buffer,self.target_buffer)])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _mbuf = [self.mt_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.mt_buffer = _mbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.mt_buffer.reverse()
                self.target_buffer.reverse()

        def lookup_token(t, d, unk_val):
            return d[t] if t in d else unk_val

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [lookup_token(f, self.source_dicts[i],
                                          self.source_unk_vals[i])
                             for (i, f) in enumerate(w.split('|'))]
                    else:
                        w = [lookup_token(w, self.source_dicts[0],
                                          self.source_unk_vals[0])]
                    tmp.append(w)
                ss_indices = tmp

                # read from mt file and map to word index
                mm = self.mt_buffer.pop()
                mm_indices = [lookup_token(w, self.target_dict,
                                           self.target_unk_val) for w in mm]
                # -----------------------------TODO: nonsense------------------#
                if self.target_vocab_size != None:
                    mm_indices = [w if w < self.target_vocab_size
                                  else self.target_unk_val
                                  for w in mm_indices]
                # -------------------------------------------------------------#

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt_indices = [lookup_token(w, self.target_dict,
                                           self.target_unk_val) for w in tt]
                #-----------------------------TODO: nonsense------------------#
                if self.target_vocab_size != None:
                    tt_indices = [w if w < self.target_vocab_size
                                    else self.target_unk_val
                                  for w in tt_indices]
                #-------------------------------------------------------------#
                source.append(ss_indices)
                mt.append(mm_indices)
                target.append(tt_indices)
                longest_source = max(longest_source, len(ss_indices))
                longest_mt = max(longest_mt, len(mm_indices))
                longest_target = max(longest_target, len(tt_indices))

                if self.token_batch_size:
                    if len(source)*longest_source > self.token_batch_size or \
                        len(mt) * longest_mt > self.token_batch_size or \
                        len(target)*longest_target > self.token_batch_size:
                        # remove last sentence pair (that made batch over-long)
                        source.pop()
                        mt.pop()
                        target.pop()
                        self.source_buffer.append(ss)
                        self.mt_buffer.append(mm)
                        self.target_buffer.append(tt)

                        break

                else:
                    if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                            len(mt) >= self.batch_size:
                        break
        except IOError:
            self.end_of_data = True

# source(batch_size, num_tokens, n_factor), uneven
# mt(batch_size, num_tokens), uneven
# target(batch_size, num_tokens), uneven
        return source, mt, target

import re
import json
import os
import sys
import random
import string
import logging
import argparse
from os.path import basename
from shutil import copyfile
from datetime import datetime
from collections import Counter
import multiprocessing
import torch
import msgpack
import pickle
import pandas as pd
import numpy as np
from QA_model.model_CoQA import QAModel
from CoQA_eval import CoQAEvaluator
from general_utils import score, BatchGen_CoQA
from general_utils import flatten_json, free_text_to_span, normalize_text, build_embedding, load_glove_vocab, pre_proc, \
    get_context_span, find_answer_span, feature_gen, token2id
from allennlp.modules.elmo import batch_to_ids
import spacy


parser = argparse.ArgumentParser(
    description='Predict using a Dialog QA model.'
)
parser.add_argument('--dev_dir', default='CoQA/')

parser.add_argument('-o', '--output_dir', default='pred_out/')
parser.add_argument('--number', type=int, default=-1, help='id of the current prediction')
parser.add_argument('-m', '--model', default='',
                    help='testing model pathname, e.g. "models/checkpoint_epoch_11.pt"')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('-bs', '--batch_size', default=1)
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--show', type=int, default=3)
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

parser.add_argument("-i", "--input", default="CoQA/dev.json")

args = parser.parse_args()
if args.model == '':
    print("model file is not provided")
    sys.exit(-1)
if args.model[-3:] != '.pt':
    print("does not recognize the model file")
    sys.exit(-1)

# create prediction output dir
os.makedirs(args.output_dir, exist_ok=True)
# count the number of prediction files
if args.number == -1:
    args.number = len(os.listdir(args.output_dir))+1
args.output = args.output_dir + 'pred' + str(args.number) + '.pckl'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def dev_eval():
    log.info('[program starts.]')
    checkpoint = torch.load(args.model)
    opt = checkpoint['config']
    opt['task_name'] = 'CoQA'
    opt['cuda'] = args.cuda
    opt['seed'] = args.seed
    if opt.get('do_hierarchical_query') is None:
        opt['do_hierarchical_query'] = False
    state_dict = checkpoint['state_dict']
    log.info('[model loaded.]')

    input_file = args.input
    vocab, test_embedding = load_dev_data(opt)
    test = build_test_data(opt, input_file, vocab)
    model = QAModel(opt, state_dict = state_dict)
    CoQAEval = CoQAEvaluator(input_file)
    log.info('[Data loaded.]')

    model.setup_eval_embed(test_embedding)

    if args.cuda:
        model.cuda()

    batches = BatchGen_CoQA(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, dialog_ctx=opt['explicit_dialog_ctx'], precompute_elmo=args.batch_size)
    sample_idx = random.sample(range(len(batches)), args.show)

    with open(input_file, "r", encoding="utf8") as f:
        dev_data = json.load(f)

    list_of_ids = []
    for article in dev_data['data']:
        id = article["id"]
        for Qs in article["questions"]:
            tid = Qs["turn_id"]
            list_of_ids.append((id, tid))

    predictions = []
    for i, batch in enumerate(batches):
        prediction = model.predict(batch)
        predictions.extend(prediction)

        if not (i in sample_idx):
            continue

        print("Story: ", batch[-4][0])
        for j in range(len(batch[-2][0])):
            print("Q: ", batch[-2][0][j])
            print("A: ", prediction[j])
            print("Gold A: ", batch[-1][0][j])
            print("---")
        print("")

    assert(len(list_of_ids) == len(predictions))
    official_predictions = []
    for ids, pred in zip(list_of_ids, predictions):
        official_predictions.append({
         "id": ids[0],
         "turn_id": ids[1],
         "answer": pred})
    with open("model_dev_prediction.json", "w", encoding="utf8") as f:
        json.dump(official_predictions, f, ensure_ascii=False)

    f1 = CoQAEval.compute_turn_score_seq(predictions).get('f1')
    em = CoQAEval.compute_turn_score_seq(predictions).get('em')
    precision = CoQAEval.compute_turn_score_seq(predictions).get('precision')
    recall = CoQAEval.compute_turn_score_seq(predictions).get('recall')
    log.warning("Test F1: {:.3f}".format(f1 * 100.0))
    log.warning("Test Precision: {:.3f}".format(precision * 100.0))
    log.warning("Test Recall: {:.3f}".format(recall * 100.0))
    log.warning("Test Exact Match: {:.3f}".format(em * 100.0))

def load_dev_data(opt): # can be extended to true test set
    with open(os.path.join(args.dev_dir, 'dev_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    vocab = meta['vocab']

    return vocab, embedding

def build_test_data(opt, dev_file, vocab):
    nlp = spacy.load('vi_spacy_model', disable=['parser'])

    # random.seed(args.seed)
    # np.random.seed(args.seed)

    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
    #                     datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    # tags
    vocab_tag = [''] + list(nlp.tagger.labels)
    # entities
    # log.info('start data preparing... (using {} threads)'.format(args.threads))

    # glove_vocab = load_glove_vocab(wv_file, wv_dim)  # return a "set" of vocabulary
    # log.info('glove loaded.')

    def proc_dev(ith, article):
        rows = []
        context = article['story']

        for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
            gold_answer = answers['input_text']
            span_answer = answers['span_text']

            answer, char_i, char_j = free_text_to_span(gold_answer, span_answer)
            answer_choice = 0 if answer == '__NA__' else \
                1 if answer == '__YES__' else \
                    2 if answer == '__NO__' else \
                        3  # Not a yes/no question

            if answer_choice == 3:
                answer_start = answers['span_start'] + char_i
                answer_end = answers['span_start'] + char_j
            else:
                answer_start, answer_end = -1, -1

            rationale = answers['span_text']
            rationale_start = answers['span_start']
            rationale_end = answers['span_end']

            q_text = question['input_text']
            if j > 0:
                q_text = article['answers'][j - 1]['input_text'] + " // " + q_text

            rows.append(
                (ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
        return rows, context


    dev, dev_context = flatten_json(dev_file, proc_dev)
    dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer', 'answer_start', 'answer_end', 'rationale',
                                    'rationale_start', 'rationale_end', 'answer_choice'])
    # log.info('dev json data flattened.')

    # print(dev)

    devC_iter = (pre_proc(c) for c in dev_context)
    devQ_iter = (pre_proc(q) for q in dev.question)
    devC_docs = [doc for doc in nlp.pipe(
        devC_iter, batch_size=64, n_threads=args.threads)]
    devQ_docs = [doc for doc in nlp.pipe(
        devQ_iter, batch_size=64, n_threads=args.threads)]

    # tokens
    devC_tokens = [[re.sub(r'_', ' ', normalize_text(w.text)) for w in doc] for doc in devC_docs]
    devQ_tokens = [[re.sub(r'_', ' ', normalize_text(w.text)) for w in doc] for doc in devQ_docs]
    devC_unnorm_tokens = [[re.sub(r'_', ' ', w.text) for w in doc] for doc in devC_docs]
    # log.info('All tokens for dev are obtained.')

    dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
    # log.info('context span for dev is generated.')

    ans_st_token_ls, ans_end_token_ls = [], []
    for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
        ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
        ans_st_token_ls.append(ans_st_token)
        ans_end_token_ls.append(ans_end_token)

    ration_st_token_ls, ration_end_token_ls = [], []
    for ration_st, ration_end, idx in zip(dev.rationale_start, dev.rationale_end, dev.context_idx):
        ration_st_token, ration_end_token = find_answer_span(dev_context_span[idx], ration_st, ration_end)
        ration_st_token_ls.append(ration_st_token)
        ration_end_token_ls.append(ration_end_token)

    dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
    dev['rationale_start_token'], dev['rationale_end_token'] = ration_st_token_ls, ration_end_token_ls

    initial_len = len(dev)
    dev.dropna(inplace=True)  # modify self DataFrame
    # log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
    # log.info('answer span for dev is generated.')

    # features
    devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, args.no_match)
    # log.info('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))
    vocab_ent = list(set([ent for sent in devC_ents for ent in sent]))

    # vocab
    dev_vocab = vocab  # tr_vocab is a subset of dev_vocab
    devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
    devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
    devQ_tokens = [["<S>"] + doc + ["</S>"] for doc in devQ_tokens]
    devQ_ids = [[2] + qsent + [3] for qsent in devQ_ids]
    # print(devQ_ids[:10])
    # tags
    devC_tag_ids = token2id(devC_tags, vocab_tag)  # vocab_tag same as training
    # entities
    devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0)  # vocab_ent same as training
    # log.info('vocabulary for dev is built.')

    prev_CID, first_question = -1, []
    for i, CID in enumerate(dev.context_idx):
        if not (CID == prev_CID):
            first_question.append(i)
        prev_CID = CID

    data = {
        'question_ids': devQ_ids,
        'context_ids': devC_ids,
        'context_features': devC_features,  # exact match, tf
        'context_tags': devC_tag_ids,  # POS tagging
        'context_ents': devC_ent_ids,  # Entity recognition
        'context': dev_context,
        'context_span': dev_context_span,
        '1st_question': first_question,
        'question_CID': dev.context_idx.tolist(),
        'question': dev.question.tolist(),
        'answer': dev.answer.tolist(),
        'answer_start': dev.answer_start_token.tolist(),
        'answer_end': dev.answer_end_token.tolist(),
        'rationale_start': dev.rationale_start_token.tolist(),
        'rationale_end': dev.rationale_end_token.tolist(),
        'answer_choice': dev.answer_choice.tolist(),
        'context_tokenized': devC_tokens,
        'question_tokenized': devQ_tokens
    }
    # with open('CoQA/test_data.msgpack', 'wb') as f:
    #     msgpack.dump(result, f)

    # log.info('saved test to disk.')
    dev = {'context': list(zip(
                        data['context_ids'],
                        data['context_tags'],
                        data['context_ents'],
                        data['context'],
                        data['context_span'],
                        data['1st_question'],
                        data['context_tokenized'])),
           'qa': list(zip(
                        data['question_CID'],
                        data['question_ids'],
                        data['context_features'],
                        data['answer_start'],
                        data['answer_end'],
                        data['rationale_start'],
                        data['rationale_end'],
                        data['answer_choice'],
                        data['question'],
                        data['answer'],
                        data['question_tokenized']))
          }
    print("test_data built")
    # embedding = torch.Tensor(meta['embedding'])
    return dev

if __name__ == '__main__':
    dev_eval()
# -*- coding:utf-8 -*-

"""
    Extractor aims to extract some infomation from the massive records.
    V1.0: mannual extract for each Bridge
"""

from web3 import Web3
import random, json, os, copy, logging, string, itertools
from easydict import EasyDict as ed
from collections import defaultdict as dd
import typing as T
from src import tools as TL, config as C, secret as S, doc_info as DOC, tracer as TCE
from src.common import my_timer, AncestorTree
from src.extractor_old import Extractor, ExtractorDL, ExtractorRule, ExtractorCompound
import openai

def _debug_should_stop(candidate_ans_str):
    return '"amount": "log[FulfilledOrder].args.takeAmount", "chain": "log[FulfilledOrder].args.order.giveChainId", "timestamp": "transaction.timeStamp", "to": "log[FulfilledOrder].args.receiverDst", "token": "log[FulfilledOrder].args.takeTokenAddress"' in candidate_ans_str

class ExtractorLLM(Extractor):
    
    # ! Open AI Python SDK: https://cookbook.openai.com/examples/assistants_api_overview_python
    # ! and https://platform.openai.com/docs/api-reference/chat/create
    # ! API refernce: https://platform.openai.com/docs/api-reference/introduction

    QUERY_TIMES_FOR_VOTING = 20

    def __init__(self, model_type:str='gpt-4o-mini', check_rpc:bool=True):
        super().__init__(check_rpc)
        f = open(C.paths().LLM_prompt_score)
        self.PROMPT = json.load(f)
        self.PRESET_KEYS = [C.PRESET_KEYS_TIMESTAMP, C.PRESET_KEYS_TOKEN, C.PRESET_KEYS_AMOUNT, C.PRESET_KEYS_TO, C.PRESET_KEYS_CHAIN,]
        f.close()
        self._load_LLM_record()
        self.LLM_records_changed = False
        self.client = None
        self.first_query_times = dd(lambda : dd(lambda: self.QUERY_TIMES_FOR_VOTING))
        self.voting_keys = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0))))) # chain => (func_name => (preset_keys => (chosen_keys by LLM => count)) ) 
        # self.voting_keys_second = dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0)))) # chain => (func_name => (preset_keys => (chosen_keys by LLM => count)) ) 
        self.first_chosen_fields = dd(lambda: dd(lambda: dd(lambda: 0))) # chain => (func_name =>  ) 
        self.second_chosen_fields = dd(lambda: dd(lambda: dd(lambda: 0))) # chain => (func_name =>  ) 
        self.final_chosen_keys = dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0)))) # chain => (role => meta_structure => {} ) 
        self.llm_chosen_fields_status = dd(lambda: dd(lambda: dd(lambda: dd(lambda : dict())))) # chain => (func_name => (preset_keys => (status) ) )
        self.unfound_answer_trxs = dd(lambda: dd(lambda: dd(list)))
        self.model_type = model_type
    
    
    def _get_client(self, model_type:str, force_reload:bool=False):
        if self.client is None or force_reload:
            proxy = json.load(open(C.paths().proxy)) if os.path.exists(C.paths().proxy) else None
            if self.client is not None:
                self.client.close()
            if model_type in ('gpt-4', 'gpt-4o', 'gpt-4o-mini'):
                self.client = openai.OpenAI(
                    # http_client=httpx.Client(proxies=proxy['http']),
                    base_url=S.secret_gpt4_url, 
                    api_key=S.secret_gpt4
                )
            elif model_type == 'llama3':
                from meta_ai_api import MetaAI
                os.environ['all_proxy']='socks5://114.212.81.12:1080'
                self.client = MetaAI(proxy=proxy)
            else:
                raise NotImplementedError
        return self.client


    def _get_client(self, model_type:str, force_reload:bool=False):
        if self.client is None or force_reload:
            proxy = json.load(open(C.paths().proxy)) if os.path.exists(C.paths().proxy) else None
            if self.client is not None:
                self.client.close()
            if model_type in ('gpt-4', 'gpt-4o', 'gpt-4o-mini'):
                self.client = openai.OpenAI(
                    # http_client=httpx.Client(proxies=proxy['http']),
                    base_url=S.secret_gpt4_url, 
                    api_key=S.secret_gpt4
                )
            elif model_type == 'llama3':
                from meta_ai_api import MetaAI
                os.environ['all_proxy']='socks5://114.212.81.12:1080'
                self.client = MetaAI(proxy=proxy)
            else:
                raise NotImplementedError
        return self.client

    def _add_LLM_record(self, response:openai.types.chat.chat_completion.ChatCompletion, input_id=None, **kwargs):
        """
            add one record to file
        """
        self.LLM_record_indexes[input_id] = len(self.LLM_records.chats)
        a = {
            'input_id':input_id,
            "output":response.choices[0].message.content,
            "in_len":response.usage.prompt_tokens,
            "out_len":response.usage.completion_tokens or TL.num_tokens_from_string(response.choices[0].message.content)
        }
        if len(kwargs): 
            a.update(kwargs)
        self.LLM_records.chats.append(a)
        self.LLM_records.statis.inputs += response.usage.prompt_tokens
        self.LLM_records.statis.outputs += response.usage.completion_tokens
    
    def __del__(self):
        # ! this is only for debugging when developping code. remove destructor when code is ready
        self._save_LLM_record()
        self._save_final_keys()

    def _meta_structure_should_drop(self, meta:str):
        return 'tokenid' in meta.lower()
    
    def _get_dependable_meta_structure(self, chain:str, trx:dict, logs,)->str:
        meta_structure = _get_meta_structure(trx, logs, True, self.meta_structures[chain]['ignore'])
        if meta_structure not in self.meta_structures[chain]['keys']: 
            # 如果不在之前选定的记录中，就选择它的祖先。
            # TODO: 后续变成LLM选择
            meta_structure = self.meta_structures[chain]['tree'].get_ancestor(meta_structure)
        if self._meta_structure_should_drop(meta_structure): 
            return ""
        return meta_structure

    def go(self, trx:dict, logs:T.List[dict], chain:str):
        """
            call this functin after `get_diverse_field`, `go_for_voting` and `verify_voting_result`
        """
        def _go(chosen_keys):
            chosen_keys_wrap = { a:[b] for a, b in chosen_keys.items()}
            extract_res = self.extract_by_rule(trx, logs, chosen_keys_wrap)
            to, token_addr, amount, timeStamp, paired_chain = (
                extract_res.get(C.PRESET_KEYS_TO, None),
                extract_res.get(C.PRESET_KEYS_TOKEN, ''),
                extract_res.get(C.PRESET_KEYS_AMOUNT, None),
                extract_res.get(C.PRESET_KEYS_TIMESTAMP, None),
                extract_res.get(C.PRESET_KEYS_CHAIN, None),
            )

            paired_chain = DOC.chain_id_to_name[C.current_bridge].get(paired_chain, '')

            to = _normalize_address_type(to)
            token_addr = _normalize_address_type(token_addr)

            token_name, token_symbol, amount = self.deal_token_and_amount(token_addr, int(amount), trx, chain, self.choose_one_rpc(chain))

            return {
                'to':to, 'token_addr':token_addr, 'token_name':token_name,
                'token_symbol':token_symbol, 'amount':amount, 'timestamp':(timeStamp), 
                'role':role, 'paired_chain':paired_chain, 'chain':chain,
                'hash':trx['hash']
            }

        func_name = trx['pureFunctionName'] or trx['methodId']
        role = DOC.function_name_to_role[C.current_bridge].get(func_name, None)
        meta_structure = self._get_dependable_meta_structure(chain, trx, logs)
        # if meta_structure in self.final_chosen_keys[chain][role]:
        chosen_keys = self.final_chosen_keys[chain][role][meta_structure]
        if chosen_keys: return _go(chosen_keys) 
        # 从邻近的meta_structure里选一个
        for other_chain in self.final_chosen_keys:
            if meta_structure in self.final_chosen_keys[other_chain][role]:
                other_keys = self.final_chosen_keys[other_chain][role][meta_structure]
                if not other_keys: continue
                try: 
                    ret = _go(other_keys)
                    return ret
                except: 
                    continue
        for other_meta_structure in self.final_chosen_keys[chain][role]:
            other_keys = self.final_chosen_keys[chain][role][other_meta_structure]
            if not other_keys: continue
            try: 
                ret = _go(other_keys)
                return ret
            except: 
                continue
        return -1 
        # else:
        #     # logging.warning(f"meta strcuture not found answer: {meta_structure}")
        #     return -2
        

    def _save_LLM_record(self):
        self.LLM_records_changed = False
        with open(C.paths().LLM_log, 'w') as f: json.dump(self.LLM_records, f, indent=1)
    
    def _load_LLM_record(self):
        p = C.paths().LLM_log
        if os.path.exists(p):
            with open(p) as f: 
                self.LLM_records = ed(json.load(f))
            self.LLM_record_indexes = {}
            for i, chat in enumerate(self.LLM_records.chats):
                self.LLM_record_indexes[chat.input_id] = i
        else:
            self.LLM_records = ed({
                'chats':[],
                'statis':{'inputs':0, 'outputs':0}
            })
            self.LLM_record_indexes:T.Dict[str, int] = {} # from "input_id"(str) to "its index within array(i.e. self.LLM_records.chats)"

    def _load_final_keys(self):
        if os.path.exists(C.paths().chosen_keys):
            f = open(C.paths().chosen_keys)
            _o = json.load(f)
            f.close()
            TL.copy_dict_to_nested_defaultdict(_o, self.final_chosen_keys)
            return True
        return False
    
    def _save_final_keys(self):
        f = open(C.paths().chosen_keys, 'w')
        json.dump(self.final_chosen_keys, f, indent=1)
        f.close()

    def _decode_LLM_responce(self, resp:str):
        try:
            obj = json.loads(resp)
            return obj
        except json.JSONDecodeError:
            return {}
    
    def reorganize_bridge_data_by_role_and_hash(self, all_bridge_data):
        ret = {}
        default_role = None
        for chain in all_bridge_data:
            if chain not in ret: 
                ret[chain] = {}
            for one_data in all_bridge_data[chain]:
                trx = one_data[0]
                if not len(trx): continue
                func_name = trx['pureFunctionName'] or trx['methodId']
                role = DOC.function_name_to_role[C.current_bridge].get(func_name, default_role)
                if role is None:
                    continue
                meta_structure:str = self._get_dependable_meta_structure(chain, trx, one_data[1:])
                if not meta_structure: 
                    continue
                if role not in ret[chain]:  ret[chain][role] = {}
                if meta_structure not in ret[chain][role]:
                    ret[chain][role][meta_structure] = {}
                ret[chain][role][meta_structure][trx['hash']] = (one_data)
        return ret


    def _filter_bridge_data_by_role_and_ts(self, all_bridge_data, chain, role, timestamp, treshold:int=3600):
        return filter_bridge_data_by_role_and_ts(all_bridge_data, chain, role, timestamp, treshold)


    def begin_voting(self, chain:str, role:str, meta_strucutre:str=''):
        # voting algorithm
        if len(self.voting_keys[chain][role][meta_strucutre]):
            return 
        if self.LLM_records_changed:
            self._save_LLM_record()
        iid = '%s.%s' % (chain, role)
        for chat in self.LLM_records.chats:
            tmp = chat['input_id'].split('.')
            if chat['input_id'].startswith(iid) and TL.md5_encryption(meta_strucutre) == chat['meta_structure']:
                try:
                    output = TL.pure_json_object(chat['output'], True)
                except:
                    output = None
                if output is None: continue
                obj = self._decode_LLM_responce(output)
                if not len(obj): continue # this response failed
                trx_hash = tmp[2]
                try:
                    for preset_key in C.PRESET_KEYS:
                        for ck, score, _ in obj[preset_key][:3]:
                            self.voting_keys[chain][role][meta_strucutre][preset_key][ck] += score
                except Exception as e:
                    logging.exception(e)

    def _handle_instance_to_check(self, trx, logs)->str:
        return TL._handle_instance_to_ask_LLM(trx, logs)

    def _pack_first_query_prompt(self, trx:dict, logs:T.List[dict], chain:str, role:str)->T.List[dict]:
        """
            The first query would add current instance to prompt (role: user)
        """
        instance_str = self._handle_instance_to_check(trx, logs)
        prompt = copy.deepcopy(self.PROMPT['first'])
        _query_sentences = ';'.join("(%d)%s" % (i, v) for i, v in enumerate(C.PRESET_KEYS_TO_PROMPT.values()))
        _query_sentences = _query_sentences.format(role='source' if role == 'dst' else 'destination')
        prompt[3]['content'] = prompt[3]['content'] % (_query_sentences, instance_str,)
        return prompt


    def _wrap_ask_LLM(self, input_id, trx, logs, chain, role, meta_structure:str):
        input_id1 = input_id
        if input_id1 not in self.LLM_record_indexes:
            prompt = self._pack_first_query_prompt(trx, logs, chain, role)
            if sum(TL.num_tokens_from_string(a['content']) for a in prompt) > C.MAX_CONTEXT_SIZE:
                # Too much token of this 
                logging.error("trx contains too much token: %s" % (trx['hash']))
                return
            # self.ask_LLM_pool.apply_async(self.ask_LLM_base, (prompt, input_id1, False, True))
            first_llm_res = self.ask_LLM_base(prompt, input_id1, False, True, meta_structure=TL.md5_encryption(meta_structure))
            self.LLM_records_changed = True
        self.first_query_times[chain][meta_structure] -= 1

    def calc_meta_structure_count(self, all_data:T.Dict[str, T.List[dict]]):
        """
            Traverse all data, to calculate the count of each meta structure.
            Less frequent meta-structure will be dropped
        """
        cache_path = os.path.join(C.paths().meta_structure)
        self.meta_structures = _init_meta_structure_trees_from_file(cache_path)
        if self.meta_structures: 
            return 

        for chain in C.chains_to_pair: 
            chain_data = all_data[chain]
            metas_cnt: T.Dict[str, int] = dd(lambda:0)
            metas_cnt_orig: T.Dict[str, int] = dd(lambda:0)
            metas_str2sets: T.Dict[str, set] = {}
            flatten_keys_cnt: T.Dict[str, int] = dd(lambda :0)
            for tx_instance in chain_data:
                # 这个循环用于统计单个log的出现次数
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:])
                for ms in meta_struct: flatten_keys_cnt[ms] += 1
            metas_relation_tree = AncestorTree()
            flatten_keys_ignore = list(m for m in flatten_keys_cnt if flatten_keys_cnt[m] < self.QUERY_TIMES_FOR_VOTING )
            for tx_instance in chain_data:
                # 过一遍数据，统计每个meta_structure的个数。 这里分原始的和经过分组的
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:], False, flatten_keys_ignore)
                meta_struct_str = ';'.join(m for m in meta_struct)
                metas_cnt_orig[meta_struct_str] += 1 
                # metas_cnt[meta_struct_str] += 1 # 如果用了ancestor，则这里不能加
                meta_struct_set = set(meta_struct)
                if meta_struct_str in metas_str2sets: continue
                metas_str2sets[meta_struct_str] = meta_struct_set
                metas_relation_tree._make_set(meta_struct_str, metas_cnt_orig[meta_struct_str])
                for _pre_str, _pre_sets in metas_str2sets.items(): 
                    if meta_struct_str == _pre_str: continue
                    if meta_struct_set == _pre_sets: continue # TODO: str 不同，但是set相同，说明它里面有相同的log， 例如`Transfer`
                    if meta_struct_set.issubset(_pre_sets): 
                        # A 是 B 的子集，证明A是B的祖先
                        metas_relation_tree.add_relation(meta_struct_str, _pre_str, metas_cnt_orig[meta_struct_str], metas_cnt_orig[_pre_str])
                    elif _pre_sets.issubset(meta_struct_set):
                        metas_relation_tree.add_relation(_pre_str, meta_struct_str, metas_cnt_orig[_pre_str],metas_cnt_orig[meta_struct_str])
            # metas_relation_tree.export_to_graphviz(f'tmp/{C.current_bridge}_{chain}')
            for tx_instance in chain_data:
                # 如果使用了ancestor，则这个循环用于统计ancestor的出现次数(后代算在ancestor头上)
                meta_struct:list = _get_meta_structure(tx_instance[0], tx_instance[1:], False, flatten_keys_ignore)
                meta_struct_str = ';'.join(m for m in meta_struct)
                all_descendants = metas_relation_tree.get_all_descendants(meta_struct_str)
                max_desc = TL.find_n_keys_with_max_values( {k: metas_cnt_orig[k] for k in all_descendants}, 10)
                for ancestor in max_desc:
                    metas_cnt[ancestor] += 1
            metas_clipped = list(itertools.filterfalse(lambda x: metas_cnt[x] <= self.QUERY_TIMES_FOR_VOTING, metas_cnt.keys()))
            trx_coverage_of_metas_clipped = sum(metas_cnt_orig[x] for x in metas_clipped)
            logging.info(f"Meta structure: chain {chain}: total {len(metas_cnt_orig)}, left {len(metas_clipped)}. Coverage: {trx_coverage_of_metas_clipped}/{len(chain_data)}={trx_coverage_of_metas_clipped/len(chain_data)}%")
            obj = {
                "keys": {x : metas_cnt[x] for x in metas_clipped},
                "tree": metas_relation_tree,
                'ignore': flatten_keys_ignore
            }
            self.meta_structures[chain] = obj 
            logging.info(f'Chain:{chain}, metas_clip:')
            for m in metas_clipped: logging.info(f"{metas_cnt_orig[m]}:{m}")

        with open(cache_path, 'w') as f:
            obj = {chain: {'keys':item['keys'], 'tree': item['tree'].to_json(), 'ignore': item['ignore'] } for chain, item in self.meta_structures.items()}
            json.dump(obj, f, indent=2)

    def go_for_voting(self, trx:dict, logs:T.List[dict], chain:str):
        """
            The first cycle, in order to get the chosen key(voted by multi queries )
        """
        if not len(trx) and not len(logs): 
            return -1, 'data_incomplete'

        func_name = trx['pureFunctionName'] or trx['methodId']
        if C.current_bridge.startswith('Across'): 
            assert 0
            default_role = 'src'
        else:
            default_role = None
        role = DOC.function_name_to_role[C.current_bridge].get(func_name, default_role)
        ancestor = self._get_dependable_meta_structure(chain, trx, logs)
        if ancestor not in self.meta_structures[chain]['keys']: 
            # This is not a common case
            # Skip it 
            return -2, 'not_in_meta_structure'
        if role is None:
            return -3, 'role_not_found'
        left_times = self.first_query_times[chain][ancestor]
        input_id = "%s.%s.%s" % (chain, role, trx['hash'])
        
        if len(self.voting_keys[chain][role][ancestor]) == len(self.PRESET_KEYS):
            # this has been voted
            return 1, ''

        if left_times > 0 and input_id not in self.LLM_record_indexes:
            self._wrap_ask_LLM(input_id, trx, logs, chain, role, ancestor)
        elif left_times <= 0:
            self.begin_voting(chain, role, ancestor)
        else:
            # This trx had been queried
            self.first_query_times[chain][ancestor] -= 1
        return 1, ''


    def _pack_assistant_string_to_prompt(self, previous_resp:str, key_to_reask:str)->T.List[T.Dict[str,str]]:
        """
            When asking the second round, pack the previous answer as "assistant" role, and 
            tell LLM that it was wrong on the key `key_to_reask`
        """
        ret = []
        ret.append({'role':'assistant', 'content':previous_resp})
        ret.append({'role':'user', 'content':self.PROMPT['second'][0]['content'] % key_to_reask })
        return ret 


    def ask_LLM_base(self, message:T.List[T.Dict[str, str]], 
                input_id=None, force_reload:bool=False, add_to_record:bool=False, 
                N:int = 1, **kwargs):
        response = TL.ask_LLM(message, self.model_type)
        # count input tokens length if is not set
        response.usage.prompt_tokens = response.usage.prompt_tokens or sum(TL.num_tokens_from_string(a['content']) for a in message)
        # count output tokens length if is not set
        response.usage.completion_tokens = response.usage.completion_tokens or (TL.num_tokens_from_string(response.choices[0].message.content))
        # add to record if needed
        if add_to_record:
            self._add_LLM_record(response, input_id, **kwargs)
        return response


    def _get_transaction_hash_set_from_LLM_records(self):
        ret = {}
        for chat in self.LLM_records['chats']:
            input_id = chat['input_id']
            t = input_id.split('.')
            chain, role, tx_hash = t[0:3]
            meta_md5 = chat['meta_structure']
            if chain not in ret: ret[chain] = {}
            if role not in ret[chain]: ret[chain][role] = {}
            if meta_md5 not in ret[chain][role]: ret[chain][role][meta_md5] = []
            if tx_hash not in ret[chain][role][meta_md5]:
                ret[chain][role][meta_md5].append(tx_hash)
        return ret

    def verify_voting_result(self, all_bridge_data:T.Dict[str, T.List[T.List[T.Dict]]]):

        if self._load_final_keys(): 
            pass

        new_bridge_data = self.reorganize_bridge_data_by_role_and_hash(all_bridge_data)
        asked_tx_hash_set = self._get_transaction_hash_set_from_LLM_records()
        GT = load_GT()

        def choose_trx_set_to_check(target_length:int, pre_tx_set:list, must_in_gt:bool=False):
            target_length = int(target_length)
            if len(choices_of_tx_hash) < target_length: 
                logging.error(f"Need to choose {target_length} from length of {len(choices_of_tx_hash)}, Returning Old records..")
                target_length = len(choices_of_tx_hash)
                # return pre_tx_set
            if must_in_gt:
                pre_chosen_tx_in_GT = list(a for a in pre_tx_set if a in GT[chain]['gt'])
            else:
                pre_chosen_tx_in_GT = pre_tx_set
            if len(pre_chosen_tx_in_GT) >= target_length : 
                return pre_chosen_tx_in_GT
            ret = copy.deepcopy(pre_chosen_tx_in_GT)
            while len(ret) < target_length: 
                new_set = random.choices(choices_of_tx_hash, k=int(target_length)-len(ret))
                if must_in_gt:
                    new_set = list(a for a in new_set if a in GT[chain]['gt'])
                ret.extend(new_set)
                ret = list(set(ret))
            return ret

        def _get_fields_sort_by_conf_scores(d:dict):
            ret = sorted(d.keys(), key = lambda x: d[x], reverse=True)
            return list( r for r in ret if (r and (r.startswith('log') or r.startswith('transaction')) ) )

        def _get_all_candidate_ans(fields_chosen_by_llm, chain, role, meta_structure):
            all_chosen_fields:T.List[str] = []
            scores_of_chosen_fields:T.Dict[str, float] = {}
            for _fld in itertools.product(*fields_chosen_by_llm):
                if len(set(_fld)) < len(_fld): continue # ! remove conflicts
                chs = dict(zip(C.PRESET_KEYS, _fld))
                chosen_fields_str = json.dumps(chs, sort_keys=True)
                all_chosen_fields.append(chosen_fields_str)
                _score = sum(self.voting_keys[chain][role][meta_structure][_k][chs[_k]] for _k in C.PRESET_KEYS )
                scores_of_chosen_fields[chosen_fields_str] = _score
            all_chosen_fields = sorted(all_chosen_fields, key = lambda x: scores_of_chosen_fields[x], reverse=True)
            return all_chosen_fields, scores_of_chosen_fields

        def _stratify_candidates(candds:list):
            layers = [5, 10, 30, 50, 100]
            ret = []
            idx = 0
            for ly in layers: 
                new_idx = int(len(candds) * ly / 100)
                ret.append(candds[idx : new_idx])
                idx = new_idx
            return ret

        def cache_checking_results(write_to_file:bool=False, obj=None):
            p = os.path.join(C.paths().candidate_chekcking_res)
            if write_to_file:
                with open(p, 'w') as f: 
                    json.dump(obj, f, indent=1)
                return 
            ret = dd(lambda: dd(lambda: dd(dict)))
            if os.path.exists(p):
                f = open(p)
                o = json.load(f)
                f.close() 
                for k1, v1 in o.items(): 
                    for k2, v2 in v1.items() : 
                        for k3, v3 in v2.items() :
                            ret[k1][k2][k3] = v3
            return ret

        def _wrap_check_one_candidate_for_tx_set(candidate_ans_str:str, target_length_to_check:int, check_pass_treshold:float, possible_answer_for_dst:str=''):
            """
                检查一个候选答案(`candidate_ans_str`)，一共检查`target_length_to_check`条记录。
                如果找到的比例大于`check_pass_treshold`, 则返回True, 否则返回False.
                `possible_answer_for_dst` 可用于加速在src上的匹配
            """
            out_of_scope_txs = []
            tx_set = []
            for checked_tx_hash in candidate_tx_checking_res[chain][role].get(candidate_ans_str, []):
                if checked_tx_hash not in tx_set: tx_set.append(checked_tx_hash)
            # checked_candidates_on_runtime[candidate_ans_str].update(tx_set)
            tx_set = choose_trx_set_to_check(target_length_to_check, tx_set, False)
            for tx_hash in tx_set:
                in_set = tx_hash in checked_candidates_on_runtime[candidate_ans_str]
                checked_candidates_on_runtime[candidate_ans_str].add(tx_hash)
                # TODO: Remove this
                # if '"to": "transaction.to"' in candidate_ans_str:
                #     candidate_ans_unfound[candidate_ans_str] += 1
                #     candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'unfound'
                #     continue
                if (tx_hash) in candidate_tx_checking_res[chain][role][candidate_ans_str]: 
                    # This has been checked before
                    if candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'out_of_scope':
                        out_of_scope_txs.append(tx_hash)
                    elif candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'found':
                        candidate_ans_found[candidate_ans_str] += (1 if not in_set else 0)
                    elif candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] == 'unfound':
                        candidate_ans_unfound[candidate_ans_str] += (1 if not in_set else 0)
                    else: assert 0
                    continue
                tx_instance = new_bridge_data[chain][role][meta_structure][tx_hash]
                trx_in_our_gt_and_scope = role == 'dst' or (tx_hash in GT[chain]['gt'] and GT[chain]['gt'][tx_hash][0] in C.chains_to_pair)
                # The first answer provided by LLM 
                with my_timer() as _tm:
                    check_res = verify_one_result(
                        all_bridge_data, chain, tx_instance, 
                        json.loads(candidate_ans_str), role == 'src', 
                        self.choose_one_rpc(chain), possible_answer_for_dst)
                    cdd_used_time_for_each_trx.append(round(_tm.get_interval(), 3))
                all_found = all(v == 'found' for k, v in check_res.items())
                out_of_scope = any(v == 'out_of_scope' for k, v in check_res.items())
                if out_of_scope or not trx_in_our_gt_and_scope:
                    out_of_scope_txs.append(tx_hash)
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'out_of_scope'
                    continue
                if not all_found:
                    candidate_ans_unfound[candidate_ans_str] += 1
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'unfound'
                else:
                    candidate_ans_found[candidate_ans_str] += 1
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx_hash] = 'found'
            if len(out_of_scope_txs) >= len(tx_set) * 0.95:
                # * it is not out-of-scope, but rather wrong in `chain` field
                for tx in out_of_scope_txs:
                    candidate_tx_checking_res[chain][role][candidate_ans_str][tx] = 'unfound'
                candidate_ans_unfound[candidate_ans_str] += len(out_of_scope_txs)
                return False
            assert (candidate_ans_found[candidate_ans_str] + candidate_ans_unfound[candidate_ans_str] 
                    == len(tx_set) - len(out_of_scope_txs) )
            acc = (candidate_ans_found[candidate_ans_str]) / (len(tx_set) - len(out_of_scope_txs))
            logging.info("candidate accuracy: %.3f" % (acc) )
            if acc >= check_pass_treshold:
                return True
            else:
                return False

        for role, chain in itertools.product(['dst', 'src'], self.voting_keys.keys()): 
            for meta_structure in self.voting_keys[chain][role]:
                if (chain in self.final_chosen_keys and role in self.final_chosen_keys[chain] and 
                        meta_structure in self.final_chosen_keys[chain][role] and 
                        (True or len(self.final_chosen_keys[chain][role][meta_structure]))  ): 
                    continue
                choices_of_tx_hash = tuple(new_bridge_data[chain][role][meta_structure].keys())
                fields_chosen_by_llm = []
                for _k in C.PRESET_KEYS:
                    fields_chosen_by_llm.append(
                        _get_fields_sort_by_conf_scores(self.voting_keys[chain][role][meta_structure][_k]))
                
                all_candidates, candidate_scores = _get_all_candidate_ans(fields_chosen_by_llm, chain, role, meta_structure)
                if not len(all_candidates):
                    logging.error(f"No candidate answer for {chain}, {role}, {meta_structure}")
                    continue
                max_score_candidate = all_candidates[0]
                # 当前面的链已经有答案时，优先检查前面的答案
                for _c in self.final_chosen_keys:
                    if  _c == chain: continue
                    if role not in self.final_chosen_keys[_c] or not len(self.final_chosen_keys[_c][role]) or meta_structure not in self.final_chosen_keys[_c][role] or not len(self.final_chosen_keys[_c][role][meta_structure]): continue
                    previous_confirmed_cdd_str=json.dumps(self.final_chosen_keys[_c][role][meta_structure], sort_keys=True)
                    if previous_confirmed_cdd_str in all_candidates and previous_confirmed_cdd_str != all_candidates[0]:
                        all_candidates.remove(previous_confirmed_cdd_str)
                        all_candidates = [previous_confirmed_cdd_str] + all_candidates
                
                # 对于一条链来说，必须先找到dst的答案
                # 然后将dst的答案用于src的筛选
                # if role == 'src':
                if False:  
                    possible_answer_for_dst = self.final_chosen_keys[chain]['dst']
                    possible_answer_for_dst = json.dumps(possible_answer_for_dst)
                else:
                    possible_answer_for_dst = ''

                all_stratified_candidates = _stratify_candidates(all_candidates)
                logging.info("Max candidate score: %.3f. Stratified scores(0): %s " % (candidate_scores[max_score_candidate],'/'.join( ('%.3f' % (candidate_scores[asc[0]] if len(asc) else 0.0)) for asc in all_stratified_candidates)))
                target_tx_set_length = len(asked_tx_hash_set[chain][role][TL.md5_encryption(meta_structure)]) * 2
                candidate_ans_unfound = dd(lambda:0)
                candidate_ans_found = dd(lambda:0) # if this is unknown
                candidate_tx_checking_res = cache_checking_results()
                checked_candidates_on_runtime = dd(set)
                possible_correct_answer = []
                
                adopted_treshold = 0.8

                for layer_num in range(len(all_stratified_candidates)):
                    wrong_candidates_this_layer = []
                    for cdd in all_stratified_candidates[layer_num]:
                        cdd_used_time_for_each_trx = []
                        with my_timer('Take %f s. {ans}'.format(ans=cdd), logging.info): 
                            res = _wrap_check_one_candidate_for_tx_set(cdd, target_tx_set_length, adopted_treshold, possible_answer_for_dst)
                        logging.info(str(cdd_used_time_for_each_trx))
                        if res and cdd not in possible_correct_answer: 
                            possible_correct_answer.append(cdd)
                            logging.info("One possible answer got for %s %s: %s. Score: %f" % (chain, role, cdd, candidate_scores[cdd]))
                            if len(possible_correct_answer) >= C.answers_wanted: 
                                break
                        elif not res:
                            wrong_candidates_this_layer.append(cdd)
                    if len(possible_correct_answer) >= C.answers_wanted: 
                        logging.info("Got enough answer(%d) for (%s,%s). Stop." % (C.answers_wanted, chain, role))
                        break
                    # lower down the treshold, and choose more transaction to be checked
                    adopted_treshold_orig = adopted_treshold
                    adopted_treshold = round(adopted_treshold * 0.9, 2)

                    target_tx_set_length *= 1.5
                    # 把错误的答案下放到后续的level中
                    for wa in wrong_candidates_this_layer: 
                        if (candidate_ans_unfound[wa] + candidate_ans_found[wa]) == 0:
                            logging.warning(f"Candidate {wa} makes no meaningful checking")
                            continue
                        prop = candidate_ans_found[wa] / (candidate_ans_unfound[wa] + candidate_ans_found[wa])
                        wa_score = prop * candidate_scores[wa] / (adopted_treshold_orig * (1))
                        target_layer = layer_num
                        for i in range(layer_num + 1, len(all_stratified_candidates)):
                            s1 = candidate_scores[all_stratified_candidates[i][-1]]
                            if wa_score >= s1: 
                                target_layer = i 
                                break
                        if target_layer != layer_num:
                            all_stratified_candidates[target_layer] = [wa] + all_stratified_candidates[target_layer]
                    cache_checking_results(True, candidate_tx_checking_res)

                if len(possible_correct_answer):
                    self.final_chosen_keys[chain][role][meta_structure] = json.loads(possible_correct_answer[0])
                else:
                    logging.error(f"no correct answer found for {chain}, {role}, {meta_structure}")
                    self.unfound_answer_trxs[chain][role][meta_structure].extend(choose_trx_set_to_check(min(10, len(choices_of_tx_hash)), [], False))
                    self.final_chosen_keys[chain][role][meta_structure] = {}
                    _tracer = TCE.get_tracer()
                    for __tx in self.unfound_answer_trxs[chain][role][meta_structure]:
                        flow = _tracer.get_trace_common(chain, __tx, 'asset')
                        if role == 'dst' and len(flow):
                            logging.error(f"DST answer unfound with asset flow: {chain}, {__tx}")
                    with open(os.path.join(C.paths().unfound_trxs), 'w') as f: 
                        json.dump(self.unfound_answer_trxs, f, indent=1)
                self._save_final_keys()

    def _pack_assistant_str(self, last_chosen_key:dict, last_extrat_res:dict):
        """
            When re-asking LLM, we need to add (virtual) answer as "assistant" prompt, 
            such that can avoid LLM to answer the wrong key. 
        """
        assistant_str = {}
        for key in C.PRESET_KEYS_TO_PROMPT:
            LLM_resp_decoded = {key: {"chosen_key":last_chosen_key[key],"value":last_extrat_res[key]}}
            assistant_str.update(LLM_resp_decoded)
        return json.dumps(assistant_str)


def filter_bridge_data_by_role_and_ts(all_bridge_data, chain, role, timestamp, treshold:int=3600):
    ret = []
    if C.current_bridge.startswith('Across'): 
        default_role = 'src'
    else:
        default_role = None
    for cur_chain in all_bridge_data:
        if cur_chain != chain: continue
        for one_data in all_bridge_data[chain]:
            trx = one_data[0]
            if not len(trx): continue
            func_name = trx['pureFunctionName'] or trx['methodId']
            r = DOC.function_name_to_role[C.current_bridge].get(func_name, default_role)
            if r != role: continue
            ts = trx.get('timestamp') or trx.get('timeStamp')
            ts, timestamp = TL.save_value_int(ts), TL.save_value_int(timestamp)
            if abs(ts - timestamp) < treshold: # 1 hour
                ret.append(one_data)
    return sorted(ret, key=lambda x: TL.save_value_int(x[0].get('timestamp', False) or x[0].get('timeStamp')))


def check_to_token_amount(check_existance:bool, af, ext_tuples, af_type, src_chain, dst_chain, w3=None):
    if not (isinstance(ext_tuples[2], str) and isinstance(ext_tuples[0], str) and isinstance(ext_tuples[1], (int, float)) ): 
        return False
    if not check_existance:
        res1 = TL.save_value_int(af.to) == TL.save_value_int(ext_tuples[0]) and \
            ( af.value == ext_tuples[1] 
                # or abs( af.value - ext_tuples[1]) / af.value  < C.HYPERPARAMETER_FEE_RATE 
             )
        if af_type == 'cash': return res1
        return res1 and TL.save_value_int(ext_tuples[2]) == TL.save_value_int(af.addr)
    else:
        if not TL.save_value_int(af.to) == TL.save_value_int(ext_tuples[0]): return False
        if not af.value or not ext_tuples[1]: return False
        if af_type == 'cash':
            return abs(af.value - ext_tuples[1]) / af.value < C.HYPERPARAMETER_FEE_RATE
        if TL.save_value_int(ext_tuples[2]) == 0: 
            return False
        dst_name, dst_symbol, dst_decimal = TL.get_token_name_symbol_decimals(af.addr, dst_chain, TL.get_rpc_endpoints(dst_chain, True, True))
        src_name, src_symbol, src_decimal = TL.get_token_name_symbol_decimals(ext_tuples[2], src_chain, TL.get_rpc_endpoints(src_chain, True, True))
        if '' in (dst_name, dst_symbol, src_name, src_symbol):
            # One of the token info getting failed 
            return False
        dst_name, src_name = dst_name.strip(), src_name.strip() 
        if TL.check_whether_token_name_and_symbol_match(src_name, src_symbol,dst_name, dst_symbol,): 
            ext_value = ext_tuples[1] / (10 ** src_decimal)
            af_value = af.value / (10 ** dst_decimal)
            return (abs(af_value - ext_value) / af_value < C.HYPERPARAMETER_FEE_RATE)
        return False


def wrap_extract_from_trace(tx_instance, extract_res, 
        chosen_fields, chain_of_dst:str, check_existance, chain_of_src:str='', w3=None):
    tracer = TCE.get_tracer()
    ret = {a:'found' for a in C.PRESET_KEYS}
    try:
        ext_amount, ext_to, ext_token = (
            TL.save_value_int(extract_res[C.PRESET_KEYS_AMOUNT]), extract_res[C.PRESET_KEYS_TO], extract_res[C.PRESET_KEYS_TOKEN])
    except KeyError:
        logging.debug("check fail reason: `to`,`token`, `amount` extract failed")
        ret[C.PRESET_KEYS_AMOUNT] = ret[C.PRESET_KEYS_TO] = ret[C.PRESET_KEYS_TOKEN] = 'unfound'
        return ret # cannot extract certain fields and values from the answer given by LLM
    ext_to = _normalize_address_type(ext_to)
    ext_token = _normalize_address_type(ext_token)
    asset_flows = tracer.get_trace_common(chain_of_dst, tx_instance[0]['hash'], 'asset')
    for af in get_final_flows(asset_flows):
        af = ed(af)
        af.value = TL.save_value_int(af.value)
        if af.value in (0, 0.0):
            continue
        if check_to_token_amount(check_existance, af, 
                                    (ext_to, ext_amount, ext_token), af.type, chain_of_src, chain_of_dst, w3):
            return {a:'found' for a in C.PRESET_KEYS} # correct (in terms of native asset)
    if not check_existance: # check on dst. For src chain, we have logged in upstream function
        logging.debug("check fail reason: `to`,`token`, `amount` check fail")
    return {a:'unfound' for a in C.PRESET_KEYS}


def _filter_dst_chain_matching_trxs(tx_instance_set:list, possible_answer_for_dst:str, callback=None):
    """
        在为src上的trx寻找匹配的dst trx时，常常会找到非常多的候选trx，但是这些trx并不都来自src chain
        这个函数会筛选掉这些不来自于`src_chain`的trx，以大幅减少要检查的trx数量
    """
    if not len(possible_answer_for_dst): 
        return tx_instance_set
    ret = []
    possible_answer_for_dst = json.loads(possible_answer_for_dst)
    for tx_instance in tx_instance_set:
        a = TL.extract_values_by_given_fields(tx_instance[0], 
            tx_instance[1:] if len(tx_instance) > 1 else [], 
            {C.PRESET_KEYS_CHAIN: [possible_answer_for_dst[C.PRESET_KEYS_CHAIN]]}, True) 
        try:
            chain_id = a[C.PRESET_KEYS_CHAIN]
            if callback(chain_id): 
                ret.append(tx_instance)
        except KeyError: 
            ret.append(tx_instance)
    return ret

def _flatten_all_keys(d:dict, cur_key:str=''):
    tmp_ret = []
    ret = []
    for subkey in d.keys():
        value = d[subkey]
        if isinstance(value, dict):
            tmp_ret.extend(_flatten_all_keys(value, subkey))
        else:
            tmp_ret.append(subkey)
    if len(cur_key) and len(tmp_ret):
        for r in tmp_ret:
            ret.append(cur_key + '.' + r)
        return ret
    else:
        return tmp_ret


def _get_meta_structure(trx:dict, logs:T.List[dict], return_str:bool=False, str_filter:list=[], return_combined_keys:bool=True):
    if len(trx):
        func_name = trx.get('pureFunctionName', None) or trx.get('methodId') or trx['input'][:10]
        all_keys = _flatten_all_keys(trx['args'], 'transaction[%s].args' % func_name)
        transaction_combined = TL.group_strings_by_hierarchy(all_keys)
        combined_keys = [transaction_combined]
    else:
        combined_keys = ['transaction[none]()']
        all_keys = []
    for log_entry in logs:
        log_keys =  _flatten_all_keys(log_entry['args'], 'log[%s].args' % (log_entry['event']))
        log_keys_combine = TL.group_strings_by_hierarchy(log_keys)
        combined_keys.append(log_keys_combine)
        all_keys.extend(log_keys)
    combined_keys = sorted(list(a for a in (combined_keys) if a not in str_filter))
    if return_combined_keys:
        if return_str: 
            return ';'.join(combined_keys)
        return combined_keys
    all_keys = sorted(list(a for a in (all_keys) if a not in str_filter))
    if return_str: 
        return ';'.join(all_keys)
    return all_keys


def verify_one_result(all_bridge_data, chain_of_tx, tx_instance, chosen_fields:dict, check_existance:bool=True, w3=None, possible_answer_for_dst:str=''):
    chosen_fields_wrap = {a:[b] for a, b in chosen_fields.items()}
    extract_res = TL.extract_values_by_given_fields(tx_instance[0], tx_instance[1:] if len(tx_instance) > 1 else [], chosen_fields_wrap, True)
    if any(v is None for v in extract_res.values()): 
        # the provided fields are wrong here  
        logging.debug("check fail reason: extract fail from trx instance")
        return {a:'unfound' for a in C.PRESET_KEYS}
    ret = {a:'found' for a in C.PRESET_KEYS}
    try:
        chain_id = extract_res[C.PRESET_KEYS_CHAIN]
    except:
        ret[C.PRESET_KEYS_CHAIN] = 'unfound'
        logging.debug("check fail reason: `chain` field")
        return ret
    chain_name = DOC.chain_id_to_name[C.current_bridge].get(chain_id, None)
    if chain_name not in C.chains_to_pair: 
        ret[C.PRESET_KEYS_CHAIN] = 'out_of_scope'
        return ret # chain not in our pairing list

    if chain_name == chain_of_tx: 
        ret[C.PRESET_KEYS_CHAIN] = 'unfound'
        logging.debug("check fail reason: `chain` field")
        return ret 
    
    if check_existance:
        ts = tx_instance[0].get('timestamp') or tx_instance[0].get('timeStamp')
        treshold = C.HYPERPARAMETER_TIME_WINDOW 
        candidate_data = filter_bridge_data_by_role_and_ts(all_bridge_data, chain_name, 'dst', ts, treshold)
        candidate_data = _filter_dst_chain_matching_trxs(candidate_data, possible_answer_for_dst, lambda x: DOC.chain_id_to_name[C.current_bridge].get(x)==chain_of_tx)
        while len(candidate_data) > 50 and treshold >= 3600 / 2: # 20 minutes
            treshold = (treshold * 2) // 3
            candidate_data = filter_bridge_data_by_role_and_ts(all_bridge_data, chain_name, 'dst', ts, treshold)
            candidate_data = _filter_dst_chain_matching_trxs(candidate_data, possible_answer_for_dst, lambda x: DOC.chain_id_to_name[C.current_bridge].get(x)==chain_of_tx)
        i = 0
        for i, cdata in enumerate(TL.mid_for(candidate_data)):
            check_status = wrap_extract_from_trace(cdata, 
                        extract_res, chosen_fields, chain_name, check_existance, chain_of_tx, w3)
            all_found = all(v == 'found' for k, v in check_status.items())
            if all_found: 
                return check_status
        for k in (C.PRESET_KEYS_TO, C.PRESET_KEYS_TOKEN, C.PRESET_KEYS_AMOUNT): 
            ret[k] = 'unfound'
        if not len(candidate_data):
            logging.debug("check fail reason: timestamp error or data not found in time period")
        else:
            logging.debug("check fail reason: `to`,`token`, `amount` check fail")
        return ret
    else:
        return wrap_extract_from_trace(tx_instance, extract_res, 
                    chosen_fields, chain_of_tx, check_existance, '', w3)


def _normalize_address_type(s:str):
    if isinstance(s, str) and not all(c in string.printable for c in s):
        return TL.normalize_address_type(TL.byte_to_hex(s, True))
    return s


def _init_meta_structure_trees_from_file(filename):
    if os.path.exists(filename):
        with open(filename) as f: obj = json.load(f)
        meta_structures = {}
        for chain, chain_data in obj.items():
            meta_structures[chain] = {
                'keys': chain_data['keys'], 
                'tree': AncestorTree.from_data(chain_data['tree']),
                'ignore': chain_data['ignore']
            }
        return meta_structures
    else:
        meta_structures = {}
    return meta_structures


def calc_net_asset_flow(asset_flows:T.List[dict]):
    """
    给定一些asset flow，计算每个地址的纯转入/转出金额。
    
    @param: `asset_flow`: List[AssetFlow]
        where `AssetFlow` is a dict defined as: 
        {
            "from": str , // optional, an wallet or contract address, and could be empty string
            "to":str,  //  an wallet or contract address
            "type": str , // one of "cash" or "token"
            "addr": str ,  // if "type" == 'token', then this is the address of token
            "value": int
        }
    """
    net_flow = dd(int)
    for flow in asset_flows:
        from_addr = flow.get("from")
        to_addr = flow["to"]
        value = flow["value"]

        if from_addr:
            net_flow[from_addr] -= value
        net_flow[to_addr] += value
    return net_flow

def get_final_flows(asset_flows):
    af_for_types = dd(list)
    for af in asset_flows:
        if af['type'] == 'token' and af.get('symbol') == 'eth': 
            af_for_types['cash'].append(af)
        else:
            af_for_types[af['type']].append(af)
    ret = []
    for _t in af_for_types:
        ret.append(af_for_types[_t][-1])
    return ret

def load_GT():
    if os.path.exists(C.paths().gt): 
        with open(C.paths().gt) as f: 
            ret = json.load(f)
    else:
        ret = {'gt':{}, 'not_in':{}, 'fail':[]}
    return ret
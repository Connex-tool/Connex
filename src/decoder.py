# -*- coding:utf-8 -*-

import typing as T
import random
import json
import os
import re
import logging, copy
import binascii
from easydict import EasyDict as ed
from collections import defaultdict as dd
# Web3 
import web3
from web3 import Web3
from hexbytes import HexBytes
from eth_utils import event_abi_to_log_topic
from web3._utils.events import get_event_data
# tools and config
from src import config as C, tools as TL, tracer as TCE

class Decoder(TL.RPCHandler):
    """
        Decode transaction and/or logs args to K-V pair
    """
    def __init__(self) -> None:
        self.init_addr_dict()
        super().__init__()

    def choose_one_rpc(self, chain: str) -> Web3:
        """
            choose (randomly) from the prepared rec endpoints in order not to 
            initiate an instance every time when called
        """
        return random.choice(self.rpcs[chain])

    def init_addr_dict(self):
        d = {
            "Ethereum":{
                "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1":"0xbfb731cd36d26c2a7287716de857e4380c73a64a",
                "0x25ace71c97b33cc4729cf772ae268934f7ab5fa1":"0x2150bc3c64cbfddbac9815ef615d6ab8671bfe43",
                "0xbeb5fc579115071764c7423a4f12edde41f106ed":"0x28a55488fef40005309e2da0040dbe9d300a64ab",
                "0x229047fed2591dbec1ef1118d64f7af3db9eb290":"0x5efa852e92800d1c982711761e45c3fe39a2b6d8",
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48":"0xa2327a938febf5fec13bacfb16ae10ecbc4cbdcf",
                "0xd38e031f4529a07996aab977d2b79f0e00656c56":"0xc72386da1f13f291f02363d0619f4d575d3dcf20",
                "0xdefa4e8a7bcba345f687a2f1456f5edd9ce97202":"0xe5e8e834086f1a964f9a089eb6ae11796862e4ce",
                "0x8a9c67fee641579deba04928c4bc45f66e26343a":"0xc8eb057f5e38f71fe42a9e59d51ac60926ec933d",
                "0x0000000000085d4780b73119b644ae5ecd22b376":"0xb650eb28d35691dd1bd481325d40e65273844f9b",
                "0x76943c0d61395d8f2edf9060e1533529cae05de6":"0x29c5c51a031165ce62f964966a6399b81165efa4",
                "0xcb9f85730f57732fc899fb158164b9ed60c77d49":"0x213062749d906f7a15156399f8dee230c28ea39a",
                "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9":"0x96f68837877fd0414b55050c9e794aecdbcfca59",
                "0x3ee18b2214aff97000d974cf647e7c347e8fa585":"0x381752f5458282d317d12c30d2bd4d6e1fd8841e", 
                "0x98f3c9e6e3face36baad05fe09d375ef1464288b":"0x3c3d457f1522d3540ab3325aa5f1864e34cba9d0", 
                "0xef4fb24ad0916217251f553c0596f8edc630eb66":"0x7ec2e51a9c4f088354ad8ad8703c12d81bf21677", 
                "0xe7351fd770a37282b91d153ee690b63579d6dd7f":"0x33b72f60f2ceb7bdb64873ac10015a35bed81717", 
                "0x5c7bcd6e7de5423a257d81b442095a1a6ced35c5":"0x90438ad3d81a0739ce1cb20c73564682388c5fdd", 
                "0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae":"0xbee13d99dd633feaa2a0935f00cbc859f8305fa7", 
                "0x3a23f943181408eac424116af7b7790c94cb97a5":"0x789c527a3a756807045ba43ddba3a5b4b65185ec", 
                "0xfc99f58a8974a4bc36e60e2d490bb8d72899ee9f":"0x5ba073f65c751f419bf880d57cf51289cbdb55fe", 
                "0x69460570c93f9de5e2edbc3052bf10125f0ca22d":"0xd16f161ca1403f55d221ea43ebf7e46b10873a06", 
            },
            'BSC':{}, 
            'Polygon':{},
            'Fantom':{}, 
            'Base':{},
            "Arbitrum":{}, 
            "Optimism":{
                "0x4200000000000000000000000000000000000007":"0xc0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d30007",
                "0x4200000000000000000000000000000000000010":"0xc0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d30010",
                "0x4200000000000000000000000000000000000016":"0xc0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d3c0d30016",
            }
        }
        d['BSC'].update(d['Ethereum'])
        d['Fantom'].update(d['Ethereum'])
        d['Polygon'].update(d['Ethereum'])
        d['Optimism'].update(d['Ethereum'])
        d['Base'].update(d['Ethereum'])
        d['Arbitrum'].update(d['Ethereum'])

        d['Base']['0xe7351fd770a37282b91d153ee690b63579d6dd7f'] = '0x979791C607a388702690599120c46332f61f592C'
        d['Base']['0x09aea4b2242abc8bb4bb78d537a67a245a7bec64'] = '0x9d7b6ea64e9faebef98170210548a777605ede87'
        d['Base']['0x1231deb6f5749ef6ce6943a275a1d3e7486f4eae'] = '0x98e3e949e8310d836a625495ea70eeaa92073862'
        d['Base']['0x3a23f943181408eac424116af7b7790c94cb97a5'] = '0x0adf13f9eb24b56b641fe2112854ce2b78c11c1a'
        d['Base']['0x5965851f21dae82ea7c62f87fb7c57172e9f2add'] = '0x369974bc46c999cb646c0fedc4970070684d13f2'
        d['Optimism']['0xe7351fd770a37282b91d153ee690b63579d6dd7f'] = '0x979791C607a388702690599120c46332f61f592C'
        self.d = d 

    def get_actual_logic_address(self, chain:str, addr:str):
        """
            Some transaction will conduct a `delegate call` from proxy contract to logic contract.
            However, the `address` field in a log entry was set to **proxy contract**, 
            which makes it challenge to decode log.
            This function will return the actual logic contract address(if it were).
            TODO: find the logic contract instead of a dict
        """
        d = self.d
        if chain in d and addr.lower() in d[chain]: 
            return d[chain][addr.lower()]
        return addr


def normalize_transaction(trx:dict):
    ret = copy.deepcopy(trx)
    if 'functionName' not in ret:
        ret['functionName'] = ''
    if ret['input'].startswith('0x00000'):
        ret['input'] = '0x' + ret['input'][10:]
    if 'methodId' not in ret:
        ret['methodId'] = ret['input'][:10]
    try:
        del ret['v']
        del ret['yParity']
    except:
        pass
    return ret

class TxDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__()
        self.skip_method = set()
        self.cache = dd(list)
        self.fail_record = dd(lambda: 0)

    def go(self, tx_obj:dict, chain:str, given_abi:T.Union[str, None]=None):
        w3 = self.choose_one_rpc(chain)
        tx_obj = normalize_transaction(tx_obj)
        if len(tx_obj['methodId']) <= 2:
            return {}
        if given_abi:
            abi = given_abi
        elif tx_obj.get('functionName', '') and 'tuple' not in tx_obj['functionName']:
            abi = TL.parse_funcname_to_abi(tx_obj['functionName'])
            # if got functionName(along with its parameters' names), then update to the DB for future use
            TL.update_signature_db_with_argname(
                {tx_obj['methodId']: { "text": tx_obj['functionName'], "full" : json.dumps(abi)}}, 
                C.paths().signatures, 'SIG_DB')
            if abi is not None and len(abi):
                abi = json.dumps([abi])
        elif 'tuple ' in tx_obj.get('functionName', ''):
            # "tuple" and a space ("tuple ") indicate there's var name 
            chosen_entry = None
            candidate_abi = TL.get_abi(self.get_actual_logic_address(chain, tx_obj.to), chain, tx_obj['methodId'], 'function')
            # bind semantics in "functionName" and the hand-maked abi
            candidate_abi = json.loads(candidate_abi)
            for abi_entry in candidate_abi:
                if abi_entry['type'] == 'function' and tx_obj['functionName'].startswith(abi_entry['name'] + "("):
                    chosen_entry = abi_entry
                    break
            if chosen_entry is not None:
                pat = r'''(?P<name>\S+)\((?P<args>(\S+ (indexed )?\S*,?)+)?\)'''
                obj = re.search(pat, tx_obj['functionName'])
                args = obj.group('args')
                args = args.split(',') if args is not None else []
                for i, ipt in enumerate(chosen_entry['inputs']):
                    if not len(ipt['name']) or ipt['name'].startswith('arg'):
                        ipt['name'] = args[i].split(' ')[1]
            abi = json.dumps(candidate_abi)
        else:
            abi = TL.get_abi(self.get_actual_logic_address(chain,tx_obj.to), chain, tx_obj['methodId'], 'function') # TODO: abi not found

        try:
            ct = w3.eth.contract(Web3.to_checksum_address(tx_obj.to), abi=abi)
            a = ct.decode_function_input(tx_obj.input)
            tx_args = ed(TL.strip_underline_of_key(a[1]))
            tx_obj.update({'args':tx_args, 'pureFunctionName':a[0].fn_name})
            return tx_obj
        except Exception as e:
            if 'Could not find any function with matching selector' in e.args[0] and given_abi is None:
                new_abi = TL.get_abi('' , 'Ethereum', tx_obj.methodId, 'function') # Force to use database ?
                if new_abi:
                    if not new_abi.startswith('['):
                        new_abi = '[' +  new_abi + ']'
                    return self.go(tx_obj, chain, new_abi)
        
        if tx_obj['methodId'] in self.skip_method:
            self.fail_record[tx_obj['methodId']] += 1
            return {}
        
        # Try every possible entry from related contracts in trace_call
        if tx_obj['methodId']in self.cache:
            call_traces = self.cache[tx_obj['methodId']]
        else:
            tracer = TCE.get_tracer()
            call_traces = tracer.get_trace_common(chain, tx_obj['hash'], 'call')
        for target_addr, input_data in call_traces:
            tmp_abi = TL.get_abi(target_addr, chain, input_data[:10], 'function')
            if tmp_abi is None or not tmp_abi: continue
            tmp_abi_obj = json.loads(tmp_abi)
            if not isinstance(tmp_abi_obj, list):
                tmp_abi = json.dumps([tmp_abi_obj])
            try:
                ct = w3.eth.contract(Web3.to_checksum_address(tx_obj.to), abi=tmp_abi)
                a = ct.decode_function_input(input_data)
                tx_args = ed(TL.strip_underline_of_key(a[1]))
                tx_obj.update({'args':tx_args, 'pureFunctionName':a[0].fn_name})
                # * TL.get_abi will do the following caching!
                # func_sigs, event_sigs = TL._encode_abi_to_signature(tmp_abi)
                # TL.update_signature_db_with_argname(func_sigs,  C.paths().signatures, 'SIG_DB')
                # TL.update_signature_db_with_argname(event_sigs, C.paths().event_signatures, 'EVENT_SIG_DB')
                self.cache[tx_obj['methodId']].append((target_addr, input_data))
                return tx_obj
            except ValueError as e:
                if 'Could not find any function with matching selector' not in e.args[0]:
                    self.fail_record[tx_obj['methodId']] += 1
                    logging.exception(e)
            except Exception as e:
                self.fail_record[tx_obj['methodId']] += 1
                logging.exception(e)
        self.skip_method.add(tx_obj['methodId'])
        self.fail_record[tx_obj['methodId']] += 1
        return {}


def _deal_nested_log_item(item:dict):
    ret = {}
    ret['indexed'] = item['indexed']
    value = item['value']
    ret['name'] = item['name']
    if isinstance(value, dict):
        ret['type'] = 'tuple'
        ret['components'] = []
        string = item['type'].strip('()')
        type_comps = string.split(',')
        i = 0
        for k, v in value.items():
            if isinstance(v, dict): 
                # TODO: ?
                a = _deal_nested_log_item(v)
                ret["components"].append({"type":"tuple","components":a})
            else:
                ret["components"].append({'type':type_comps[i], "name":k})
            i += 1
    else:
        ret['type'] = item['type']
        ret['internalType'] = item['type']
    
    return ret


class LogDecoder(Decoder):

    def __init__(self) -> None:
        super().__init__()
        self.skip_event = set()
        self.fail_record = dd(lambda: 0)

    def decode_log(self,result: T.Dict[str, T.Any], abi:T.List[T.Dict], contract_instance):
        data = [t[2:] for t in result['topics']]
        data += [result['data'][2:]]
        data = HexBytes("0x" + "".join(data))
        selector = data[:32]
        if not isinstance(abi, list):
            abi = [abi]
        abis_by_sig = {event_abi_to_log_topic(item) : item for item in abi if item['type'] == 'event'}
        func_abi = abis_by_sig.get(selector, None)
        if func_abi is None:
            new_abi = TL.get_abi('' ,'Ethereum', result['topics'][0], 'event') # Force to get data from dataabse
            if new_abi: 
                a = json.loads(new_abi)
                if not isinstance(a, list): 
                    a = [a]
                return self.decode_log(result, a, contract_instance)
            else:
                self.fail_record[binascii.b2a_hex(selector)] += 1
                logging.warning(
                    "The abi doesn't contain the selector:{};addr:{};tx_hash:{}".format(
                    binascii.b2a_hex(selector), result['address'], result['transactionHash']))
            return {}
        new_result = copy.deepcopy(result)
        if isinstance(new_result['topics'][0], str):
            new_result['topics'] = list(map(lambda x: binascii.a2b_hex(x[2:]) , new_result['topics']))
        # NOTE: 这里可能会由于一些不适用utf-8编码导致错误
        try:
            ret = get_event_data(contract_instance.w3.codec, func_abi, new_result)
        except UnicodeDecodeError:
            logging.error(
                "String decode fail due to non-unicode encoding.Addr:{};log_index:{};tx_hash:{}".format(
                result['address'], result['logIndex'], result['transactionHash']))
            return {}
        return ret

    def go(self, log_entries:T.List[T.Dict[str, T.Any]], chain:str):
        logs_dict = []
        for log_entry in log_entries:
            if not len(log_entry['topics']): 
                continue
            no_correct_abi = False
            addr = self.get_actual_logic_address(chain, log_entry['address'])
            contract = self.choose_one_rpc(chain).eth.contract(Web3.to_checksum_address(addr))
            abi_of_log = TL.get_abi(addr, chain, log_entry['topics'][0], 'event')
            if not abi_of_log or not len(abi_of_log):
                # logging.warning("ABI gettting fail! log obj:" + str(log_entry))
                # logs_dict.append({})
                # continue
                no_correct_abi = True 
            if not no_correct_abi:
                abi_of_log = json.loads(abi_of_log)
                try:
                    _tmp = self.decode_log(log_entry, abi_of_log, contract)
                    if len(_tmp):
                        _tmp['args'] = TL.strip_underline_of_key(_tmp['args'])
                        logs_dict.append(_tmp)
                        continue
                    no_correct_abi = True
                except Exception as e:
                    logging.error("Error handling log(%s). Trx hash:%s, logindex:%s" % 
                                (str(e), log_entry['transactionHash'], log_entry['logIndex']))
                    # logging.exception(e)
                    no_correct_abi = True
            if log_entry['topics'][0] in self.skip_event:
                self.fail_record[log_entry['topics'][0]] += 1
                continue
            if no_correct_abi:
                tracer = TCE.get_tracer()
                log_traces = tracer.get_trace_common(chain, log_entry['transactionHash'], 'log')
                found_sigs = False 
                for log_trace in log_traces:
                    if log_trace['topics'][0] != log_entry['topics'][0]:
                        continue
                    addr = log_trace['address']
                    new_abi_from_trace_str = TL.get_abi(addr, chain, log_trace['topics'][0], 'event')
                    if not new_abi_from_trace_str:
                        # NOT FOUND
                        continue
                    if isinstance(new_abi_from_trace_str, str):
                        new_abi_from_trace = json.loads(new_abi_from_trace_str)
                    else:
                        new_abi_from_trace = new_abi_from_trace_str
                    try:
                        _tmp = self.decode_log(log_entry, new_abi_from_trace, contract)
                        if len(_tmp):
                            _tmp['args'] = TL.strip_underline_of_key(_tmp['args'])
                            logs_dict.append(_tmp)
                            _, event_sig = TL._encode_abi_to_signature(new_abi_from_trace_str)
                            TL.update_signature_db_with_argname(event_sig, C.paths().event_signatures, 'EVENT_SIG_DB')
                            found_sigs = True
                            break
                    except web3.exceptions.LogTopicError: 
                        pass
                    except Exception as e:
                        logging.error(str(e))
                if not found_sigs:
                    self.skip_event.add(log_entry['topics'][0])
                    self.fail_record[log_entry['topics'][0]] += 1

        return logs_dict



from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from trajectory_bench.config import load_config, safe_config, save_json
from trajectory_bench.llm import APIError, ChatClient
from trajectory_bench.workflow import lock_run, score_packet, write_run_reports


class Response(io.BytesIO):
    status = 200


class Opener:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def open(self, req, timeout):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return Response(self.response)


def conf(folder):
    secret = Path(folder) / 'key.txt'
    secret.write_text('TEST-SECRET-DO-NOT-LOG', encoding='utf-8')
    return {'provider': 'test', 'base_url': 'https://provider.invalid/v1',
            'model': 'test-model', 'api_key_file': str(secret), 'max_attempts': 1}


def evidence():
    return {'domain': 'airline', 'policy': 'Reply accurately.', 'available_tool_schemas': [],
            'history': [{'i': 0, 'ref': 'h:0', 'role': 'user', 'content': 'Tell me the policy.'},
                        {'i': 1, 'ref': 'h:1', 'role': 'assistant', 'customer_visible': True,
                         'content': 'Reply accurately.'}], 'actual_execution': []}


def annotation():
    scores = {d: {'status': 'pass', 'reason': 'Accurate reply', 'refs': ['h:0','h:1'],
                  'requirement_ids': ['Q1'], 'issue_ids': []}
              for d in ('RULE','CONSENT','OUTCOME','REPORT','FACT','STEPS')}
    return {'groups': [{'group_id':'G1','goal':'Policy explanation','object':'policy',
                        'request_refs':['h:0'],'revisions':[],
                        'requirements':[{'id':'Q1','content':'Accurate reply','basis':'policy','refs':['h:0']}],
                        'scores':scores,'end_state':'Reply sent'}],
            'issues':[],'evidence_links':[],'auxiliary_notes':[]}


class TransportTests(unittest.TestCase):
    def test_exact_request_cache_prevents_resampling_and_logs_no_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            c=conf(folder)
            body=json.dumps({'model':'actual-test-model','choices':[{'message':{'content':'{}'}}]}).encode()
            opener=Opener(body)
            client=ChatClient(c,Path(folder)/'audit')
            with patch('urllib.request.build_opener',return_value=opener):
                first=client.call([{'role':'user','content':'Hello'}],request_id='r1')
                second=client.call([{'role':'user','content':'Hello'}],request_id='r1')
                with self.assertRaises(APIError):
                    client.call([{'role':'user','content':'Changed'}],request_id='r1')
            self.assertEqual(first,second)
            self.assertEqual(opener.calls,1)
            for p in (Path(folder)/'audit').rglob('*.json'):
                self.assertNotIn('TEST-SECRET-DO-NOT-LOG',p.read_text(encoding='utf-8'))
            receipt=json.loads((Path(folder)/'audit/r1/receipt.json').read_text())
            self.assertEqual(receipt['actual_model'],'actual-test-model')
            self.assertFalse(receipt['model_identity_verified'])

    def test_malformed_http200_is_terminal_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            opener=Opener(b'NOT JSON')
            client=ChatClient(conf(folder),Path(folder)/'audit')
            with patch('urllib.request.build_opener',return_value=opener):
                for _ in range(2):
                    with self.assertRaises(APIError):
                        client.call([],request_id='r1')
            self.assertEqual(opener.calls,1)

    def test_http_auth_error_does_not_retry_or_leak_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            c=conf(folder);c['max_attempts']=3
            error=urllib.error.HTTPError(c['base_url'],401,'Unauthorized',{},io.BytesIO(b'TEST-SECRET-DO-NOT-LOG'))
            opener=Opener(error)
            with patch('urllib.request.build_opener',return_value=opener):
                with self.assertRaises(APIError):
                    ChatClient(c,Path(folder)/'audit').call([],request_id='r1')
            self.assertEqual(opener.calls,1)
            saved=(Path(folder)/'audit/r1/attempt_01.json').read_text()
            self.assertIn('[REDACTED]',saved)
            self.assertNotIn('TEST-SECRET-DO-NOT-LOG',saved)

    def test_config_resolves_private_key_reference_and_redacts_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            save_json(path,{'judge':{'model':'x','base_url':'https://provider.invalid', 'api_key_file':'key.txt'}})
            cfg=load_config(path)
            self.assertEqual(cfg['judge']['api_key_file'],str((path.parent/'key.txt').resolve()))
            self.assertNotIn('api_key_file',safe_config(cfg)['judge'])
            bad=copy.deepcopy(cfg);bad['judge']['api_key']='unsafe'
            save_json(path,bad)
            with self.assertRaises(ValueError):load_config(path)


class WorkflowTests(unittest.TestCase):
    def test_score_report_resume_does_not_repeat_judge(self):
        calls=[]
        class Judge:
            def __init__(self,*args):pass
            def call(self,messages,request_id):
                calls.append(messages)
                return {'model':'test','choices':[{'message':{'content':json.dumps(annotation())}}]}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            cfg={'judge':{'model':'test','base_url':'https://provider.invalid','api_key_env':'TEST'}}
            packet=evidence()
            row=score_packet('c1',packet,cfg,out,client_factory=Judge)
            again=score_packet('c1',packet,cfg,out,client_factory=Judge)
            self.assertEqual(row,again)
            self.assertEqual(len(calls),1)
            self.assertEqual(row['automatic']['status'],'scored')
            self.assertEqual(row['automatic']['verdict'],'pass')
            write_run_reports([row],out)
            self.assertTrue((out/'report.html').is_file())
            self.assertTrue((out/'records.jsonl').is_file())

    def test_malformed_judge_stays_unscored_and_retains_text(self):
        class Judge:
            def __init__(self,*args):pass
            def call(self,*args,**kwargs):
                return {'choices':[{'message':{'content':'BAD RESPONSE'}}]}
        with tempfile.TemporaryDirectory() as folder:
            row=score_packet('c1',evidence(),{'judge':{'model':'test'}},Path(folder),client_factory=Judge)
            self.assertEqual(row['automatic']['status'],'technical_unscored')
            self.assertIsNone(row['automatic']['verdict'])
            self.assertEqual(row['raw_text'],'BAD RESPONSE')

    def test_configuration_change_cannot_overwrite_locked_run(self):
        with tempfile.TemporaryDirectory() as folder:
            lock_run(Path(folder),{'seed':1},{'tasks':[1]})
            with self.assertRaises(ValueError):lock_run(Path(folder),{'seed':2},{'tasks':[1]})

    def test_incomplete_source_never_calls_judge(self):
        class ForbiddenJudge:
            def __init__(self,*args):raise AssertionError('Incomplete source must not invoke judge')
        with tempfile.TemporaryDirectory() as folder:
            packet=evidence();packet['capture_integrity']={'episode_status':'error'}
            row=score_packet('c1',packet,{'judge':{'model':'test'}},Path(folder),client_factory=ForbiddenJudge)
            self.assertEqual(row['technical_error']['type'],'IncompleteEpisode')
            self.assertEqual(row['automatic']['status'],'technical_unscored')


if __name__=='__main__':unittest.main()

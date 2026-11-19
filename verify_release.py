"""Offline package verification. Never calls a model endpoint or new business task.

The optional native smoke uses a mock environment and scripted model responses;
it is an interface test, not a new real-model experiment.
"""
from __future__ import annotations

from pathlib import Path
import argparse,atexit,contextlib,hashlib,io,json,os,socket,sys,tempfile,unittest
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['LITELLM_TELEMETRY']='False'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results/offline_verification')
    parser.add_argument('--tests',action='store_true',help='Run offline unit/native-scripted tests; optional tau2 deps needed for native tests')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(out/'tmp');Path(tempfile.tempdir).mkdir(exist_ok=True)
    from trajectory_bench.config import load_json,save_json,load_config
    from trajectory_bench.evidence import load_evidence
    from trajectory_bench.grading import aggregate
    from trajectory_bench.cli import main as cli_main
    from trajectory_bench.native import NativeRunSpec,prepare_backend
    blocked=[];original_connect=socket.socket.connect
    # LiteLLM registers an async-client exit hook even when no clients were
    # created. On restricted Windows hosts its fresh event loop can block in
    # socketpair during interpreter shutdown. Track that one hook and remove
    # it only in this network-blocked verifier, which exercises no live model
    # transport. LiteLLM may allocate unused client objects during import.
    original_register=atexit.register;async_cleanup_callbacks=[]
    def track_exit_hook(callback,*positional,**keywords):
        if (getattr(callback,'__module__','')=='litellm.llms.custom_httpx.async_client_cleanup'
                and getattr(callback,'__name__','')=='cleanup_wrapper'):
            async_cleanup_callbacks.append(callback)
        return original_register(callback,*positional,**keywords)
    atexit.register=track_exit_hook
    def no_network(sock,address):
        blocked.append(str(address));raise AssertionError('Network forbidden in offline verifier')
    socket.socket.connect=no_network
    report={'new_model_calls':0,'new_business_trials':0,'native_scripted_tests_are_not_model_experiments':True}
    try:
        tasks=load_json(HERE/'src/trajectory_bench/resources/task_index.json')
        report['fixed_tasks']=len(tasks);assert len(tasks)==45
        config=load_config(HERE/'examples/config.example.json')
        assert Path(config['native']['backend_root']).resolve()==(HERE/'vendor/tau2-bench').resolve()
        assert Path(config['native']['data_dir']).resolve()==(HERE/'vendor/tau2-bench/data').resolve()
        prepare_backend(NativeRunSpec('airline',['21'],'offline-agent','offline-user',backend_root=HERE/'vendor/tau2-bench'))
        import tau2
        assert Path(tau2.__file__).resolve().is_relative_to(HERE/'vendor/tau2-bench')
        from tau2.data_model.simulation import TextRunConfig
        from tau2.multilingual.runtime import load_run_language
        coverage={}
        for domain in ['airline','retail','telecom']:
            ids=[str(r['task_id']) for r in tasks if r['domain']==domain]
            cfg=TextRunConfig(domain=domain,language='en',task_ids=ids,agent='llm_agent',user='user_simulator',
                              llm_agent='offline-agent',llm_user='offline-user',task_split_name='base')
            loaded=load_run_language(cfg)
            available={str(t.id) for t in loaded.tasks}
            assert set(ids)<=available
            coverage[domain]={'selected':len(ids),'catalog_loaded':len(available),'missing':[]}
        report['task_catalogs']=coverage
        reference=HERE/'reference_data'
        checks=load_json(reference/'checksums.json')
        assert all(hashlib.sha256((reference/name).read_bytes()).hexdigest()==digest for name,digest in checks.items())
        report['frozen_reference_files_hash_verified']=len(checks)
        annotations=[json.loads(x) for x in (reference/'reference/annotations.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        summary=aggregate(annotations)
        assert summary==load_json(reference/'statistics/reproduced_summary.json')
        save_json(out/'reference540.json',summary)
        report['reference_aggregate']=summary
        total_history=total_events=0
        for p in sorted((reference/'inputs').glob('tr*.json')):
            packet=load_evidence(p)
            total_history+=len(packet['history']);total_events+=len(packet['actual_execution'])
        report['normalized_inputs_imported']=len(list((reference/'inputs').glob('tr*.json')))
        report['history_entries']=total_history;report['execution_events']=total_events
        replay=HERE/'examples/replay/retail100'
        raw=load_evidence(replay/'cases/run0001/episode.json')
        save_json(out/'retail100_converted.json',raw)
        assert len(raw['history'])==54 and len(raw['actual_execution'])==11
        report['real_record_conversion']={'history':54,'events':11,'new_model_inference':False}
        streams=io.StringIO()
        with contextlib.redirect_stdout(streams):
            code=cli_main(['report','--records',str(replay/'records.jsonl'),'--reviews',str(replay/'source_reviews.jsonl'),'--output',str(out/'replay_report')])
        assert code==0,streams.getvalue()
        view=load_json(out/'replay_report/report.json')
        assert view['records'][0]['automatic']['verdict']=='fail'
        assert view['records'][0]['reviewed']['verdict']=='fail'
        assert view['records'][0]['automatic']['annotation']!=view['records'][0]['reviewed']['annotation']
        report['saved_report_and_review_overlay']='passed; raw automatic output retained, correction separate'
        if args.tests:
            result_stream=io.StringIO()
            tests=unittest.TestLoader().discover(str(HERE/'tests'))
            result=unittest.TextTestRunner(stream=result_stream,verbosity=2).run(tests)
            (out/'unit_tests.log').write_text(result_stream.getvalue(),encoding='utf-8')
            report['unit_tests']={'run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped)}
            assert result.wasSuccessful(),result_stream.getvalue()[-1000:]
        report['network_attempts']=len(blocked)
        assert not blocked,'Unexpected attempted network access'
        litellm_module=sys.modules.get('litellm')
        clients=getattr(getattr(litellm_module,'in_memory_llm_clients_cache',None),'cache_dict',{})
        report['async_client_exit_cleanup']={'cached_client_objects':len(clients),
            'unused_callbacks_removed':len(async_cleanup_callbacks),'live_model_transport_used':False,
            'scope':'Offline verifier only; import-created clients had no live transport; normal task cleanup is unchanged'}
        report['status']='passed'
        save_json(out/'verification.json',report)
        print(json.dumps({'status':'passed','tasks':45,'reference_trajectories':540,'api_calls':0,'tests':report.get('unit_tests')},ensure_ascii=False))
        return 0
    finally:
        atexit.register=original_register
        # Even on a failing assertion, this command never performs a live model
        # request. Its import-created clients need no new event loop at exit.
        for callback in async_cleanup_callbacks:atexit.unregister(callback)
        socket.socket.connect=original_connect


if __name__=='__main__':raise SystemExit(main())

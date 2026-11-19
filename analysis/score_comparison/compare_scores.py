"""Read-only paired comparison of the first GLM output and final source reviews.

No API clients, semantic label changes or experimental modules are imported.
All newly generated files belong to --out; source files are hash checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
for _parent in Path(__file__).resolve().parents:
    _candidates=[_parent/"src",_parent/"process_bench/src"]
    _found=next((p for p in _candidates if (p/"trajectory_bench/grading.py").is_file()),None)
    if _found is not None:
        sys.path.insert(0,str(_found));break
from trajectory_bench.grading import aggregate, validate_annotation

MODEL_NAMES = {"qwen9b":"Qwen3.5-9B", "llama8b":"Llama3.1-8B", "ds_flash":"DeepSeek Flash", "glm_flash":"GLM-5.3-Flash"}
DOMAIN_NAMES = {"retail":"零售", "airline":"航空", "telecom":"电信"}
MODEL_ORDER = ["qwen9b","llama8b","ds_flash","glm_flash"]
DOMAIN_ORDER = ["retail","airline","telecom"]
DIMS = ["RULE","CONSENT","OUTCOME","REPORT","FACT","STEPS"]
DIM_NAMES = dict(zip(DIMS,["规则与对象","用户确认","实际结果","完成说明","事实说明","必要步骤"]))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarized_annotation(case_id, annotation):
    groups=[{"group_id":g["group_id"],"scores":{d:{"status":g["scores"][d]["status"]} for d in DIMS}}
            for g in annotation["groups"]]
    val={"case_id":case_id,"groups":groups}
    summary=aggregate([val])
    states=summary["trajectories"]
    final=next(s for s in ["pass","fail","unknown","na"] if states[s])
    dimensions={d:next(s for s in ["pass","fail","unknown","na"] if summary["dimensions"][d][s]) for d in DIMS}
    return val,final,dimensions


def comparison(rows, level, key, total_count):
    auto=aggregate([x["automatic_states"] for x in rows])["trajectories"]
    review=aggregate([x["reviewed_states"] for x in rows])["trajectories"]
    agreement=sum(x["automatic_verdict"]==x["reviewed_verdict"] for x in rows)
    changes=Counter(f'{x["automatic_verdict"]}->{x["reviewed_verdict"]}' for x in rows)
    n=len(rows)
    label=MODEL_NAMES.get(key,DOMAIN_NAMES.get(key,key))
    return {"level":level,"key":key,"label":label,"total_records":total_count,"paired_complete":n,
            "technical_unscored":total_count-n,"automatic_pass":auto["pass"],"automatic_fail":auto["fail"],
            "automatic_unknown":auto["unknown"],"automatic_na":auto["na"],
            "automatic_rate_denominator":auto["pass"]+auto["fail"],"automatic_pass_rate":auto["pass_rate_decided"],
            "reviewed_pass":review["pass"],"reviewed_fail":review["fail"],"reviewed_unknown":review["unknown"],
            "reviewed_na":review["na"],"reviewed_rate_denominator":review["pass"]+review["fail"],
            "reviewed_pass_rate":review["pass_rate_decided"],
            "review_minus_auto_percentage_points":100*(review["pass_rate_decided"]-auto["pass_rate_decided"]),
            "verdict_agree":agreement,"verdict_disagree":n-agreement,"agreement":agreement/n if n else None,
            "automatic_pass_to_reviewed_fail":changes["pass->fail"],
            "automatic_fail_to_reviewed_pass":changes["fail->pass"]}


def ranking(rows, side):
    field=side+"_pass_rate"
    ordered=sorted(rows,key=lambda x:(-x[field],x["key"]))
    return [{"model":x["key"],"name":MODEL_NAMES.get(x["key"],x["key"]),
             "rank":1+sum(y[field]>x[field] for y in rows),"pass_rate":x[field]} for x in ordered]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--auto-scores","--raw-scores",dest="auto_scores",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--run-index",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--hash-lock",type=Path)
    parser.add_argument("--api-config",type=Path)
    parser.add_argument("--source-info",type=Path,help="Portable frozen source metadata; default beside raw-scores directory")
    parser.add_argument("--evidence-root",type=Path,help="Root containing run-index input_file paths; default run-index parent")
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    data=root/"02_论文初稿/交互轨迹过程可靠性基准_v6_配套材料"
    original=root/"experiments/trajectory_process_full540_20261110"
    source_review=root/"experiments/trajectory_process_source_completion_20261111"
    auto_dir=args.auto_scores or original/"scores"
    reference_path=args.reference or data/"reference/annotations.jsonl"
    index_path=args.run_index or data/"run_index.json"
    manifest_path=args.manifest or original/"manifest.json"
    lock_path=args.hash_lock or source_review/"RAW_GLM_COMPARISON_LOCK.json"
    config_path=args.api_config or original/"API_CONFIG.json"
    info_path=args.source_info or auto_dir.parent/"comparison_sources.json"
    portable=read(info_path) if info_path.exists() else None
    # The portable source manifest inherits the frozen original hashes and
    # minimum public run metadata; it contains neither API keys nor labels.
    if portable is not None:
        locked=portable.get("published_score_hashes", portable["original_score_hashes"])
        manifest={r["case_id"]:r for r in portable["manifest_cases"]}
        api_config=portable["api_config_safe"]
        source_fingerprints=portable["original_source_fingerprints"]
    else:
        # Locks contain old absolute paths; compare immutable basenames.
        locked={Path(name).name:value for name,value in read(lock_path)["files"].items()}
        manifest={r["case_id"]:r for r in read(manifest_path)["cases"]}
        api_config=read(config_path)
        source_fingerprints={"manifest_sha256":sha(manifest_path),"api_config_sha256":sha(config_path),"hash_lock_sha256":sha(lock_path)}
    index=read(index_path)
    refs={r["case_id"]:r for r in (json.loads(line) for line in reference_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    assert len(index)==len(refs)==len(manifest)==len(locked)==540
    assert len({x["case_id"] for x in index})==540
    before_hashes={p.name:sha(p) for p in auto_dir.glob("tr*.json")}
    assert before_hashes==locked,"Distributed automatic outputs differ from the release hash manifest"
    pairs=[];technical=[];settings=Counter();all_meta=[];compatibility=[];final_compatibility=[]
    evidence_root=args.evidence_root or index_path.parent
    for meta in index:
        cid=meta["case_id"];m=manifest[cid]
        for field in ["domain","model_alias","task_id","repetition"]:
            assert str(meta[field])==str(m[field]),(cid,field)
        raw=read(auto_dir/(cid+".json"))
        assert raw["case_id"]==cid
        assert raw["input_sha256"]==meta["input_sha256"]==m["input_sha256"]
        settings[(raw.get("provider"),raw.get("actual_model"))]+=1
        reviewed,rv,rd=summarized_annotation(cid,refs[cid])
        evidence_path=evidence_root/meta["input_file"]
        assert sha(evidence_path)==meta["input_sha256"]
        evidence=read(evidence_path)
        final_validation=validate_annotation(refs[cid],evidence)
        final_compatibility.append({"case_id":cid,"model_alias":meta["model_alias"],"domain":meta["domain"],
                                   "valid":final_validation["valid"],"errors":final_validation["errors"],"warnings":final_validation["warnings"]})
        record={"case_id":cid,"model_alias":meta["model_alias"],"domain":meta["domain"],
                "task_id":meta["task_id"],"repetition":meta["repetition"],"original_reward":meta["original_reward"],
                "published_score_sha256":before_hashes[cid+".json"],
                "original_score_sha256":(portable["original_score_hashes"][cid+".json"] if portable else before_hashes[cid+".json"]),"input_sha256":meta["input_sha256"],
                "reviewed_verdict":rv,"reviewed_dimensions":rd,"reviewed_states":reviewed}
        all_meta.append(record)
        parsed=raw.get("parsed")
        valid=not raw.get("validation_errors") and isinstance(parsed,dict) and bool(parsed.get("groups"))
        if valid:
            automatic,av,ad=summarized_annotation(cid,parsed)
            record.update(automatic_status="complete",automatic_verdict=av,automatic_dimensions=ad,automatic_states=automatic)
            validation=validate_annotation(parsed,evidence)
            compatibility.append({"case_id":cid,"model_alias":meta["model_alias"],"domain":meta["domain"],
                                  "original_protocol_complete":True,"release_validator_valid":validation["valid"],
                                  "release_validation_errors":validation["errors"],"release_validation_warnings":validation["warnings"]})
        else:
            record.update(automatic_status="technical_unscored",automatic_verdict=None,
                          automatic_dimensions=None,automatic_states=None,validation_errors=raw.get("validation_errors"))
            technical.append({k:record[k] for k in ["case_id","model_alias","domain","task_id","repetition","validation_errors","reviewed_verdict"]})
        pairs.append(record)
    valid=[x for x in pairs if x["automatic_status"]=="complete"]
    assert len(valid)==527 and len(technical)==13
    overall=comparison(valid,"all","all",540)
    assert overall["verdict_agree"]==459
    model_rows=[comparison([x for x in valid if x["model_alias"]==m],"model",m,135) for m in MODEL_ORDER]
    domain_rows=[comparison([x for x in valid if x["domain"]==d],"domain",d,180) for d in DOMAIN_ORDER]
    cell_rows=[comparison([x for x in valid if x["model_alias"]==m and x["domain"]==d],"model_domain",m+"/"+d,45)
               for d in DOMAIN_ORDER for m in MODEL_ORDER]
    final_all=aggregate([x["reviewed_states"] for x in pairs])
    assert {k:final_all["trajectories"][k] for k in ["pass","fail","unknown"]}=={"pass":291,"fail":220,"unknown":29}
    assert final_all["business_groups"]["n"]==874
    transitions=Counter(f'{x["automatic_verdict"]}->{x["reviewed_verdict"]}' for x in valid)
    dimensions=[]
    for dim in DIMS:
        matrix=Counter(f'{x["automatic_dimensions"][dim]}->{x["reviewed_dimensions"][dim]}' for x in valid)
        dimensions.append({"dimension":dim,"name":DIM_NAMES[dim],"n":527,
                           "agreement":sum(v for k,v in matrix.items() if k.split('->')[0]==k.split('->')[1])/527,
                           "changed":sum(v for k,v in matrix.items() if k.split('->')[0]!=k.split('->')[1]),
                           **{f'count_{a}_to_{b}':matrix[a+'->'+b] for a in ["pass","fail","unknown","na"] for b in ["pass","fail","unknown","na"]}})
    rank_all={side:ranking(model_rows,side) for side in ["automatic","reviewed"]}
    rank_scenes={d:{side:ranking([dict(x,key=x["key"].split('/')[0]) for x in cell_rows if x["key"].endswith('/'+d)],side)
                       for side in ["automatic","reviewed"]} for d in DOMAIN_ORDER}
    providers=[]
    for (provider,model),n in settings.items():
        opts=api_config["providers"][provider]["options"]
        providers.append({"provider":provider,"actual_model":model,"records":n,"configured_model":opts["model"],
                          "temperature":opts.get("temperature"),"thinking":opts.get("thinking",opts.get("enable_thinking")),
                          "reasoning_effort":opts.get("reasoning_effort")})
    same_ranking=[x["model"] for x in rank_all["automatic"]]==[x["model"] for x in rank_all["reviewed"]]
    summary={"created_at":datetime.now(timezone.utc).isoformat(),"complete":True,
             "origin":"First 2026-11-10 GLM automatic outputs compared against final540 source-evidence review; no resampling",
             "interpretation":"Agreement with assistant/model-assisted source review; not independent human accuracy or new-task validation",
             "cohort":"Same527 complete original outputs on both sides; original13 format failures excluded from score comparison and separately counted",
             "rate_definition":"pass/(pass+fail), unknown and NA excluded separately on each side; paired cohort same but decided denominators can differ",
             "model_settings":providers,"overall":overall,"by_model":model_rows,"by_domain":domain_rows,
             "by_model_domain":cell_rows,"transitions":dict(transitions),"rankings_all":rank_all,
             "same_overall_model_order":same_ranking,"rankings_by_domain":rank_scenes,
             "reviewed_all540_separate":final_all,"dimension_agreement":dimensions,
             "technical_cases":technical,"new_api_calls":0,"new_business_runs":0}
    compatibility_summary={"n_original_complete":len(compatibility),
             "current_release_valid":sum(x["release_validator_valid"] for x in compatibility),
             "current_release_invalid":sum(not x["release_validator_valid"] for x in compatibility),
             "current_release_warning_cases":sum(bool(x["release_validation_warnings"]) for x in compatibility),
             "interpretation":"Mechanical compatibility against current release validator only; does not replace original protocol complete/invalid classification or verify semantic correctness",
             "error_case_ids":[x["case_id"] for x in compatibility if not x["release_validator_valid"]]}
    summary["release_validator_compatibility"]=compatibility_summary
    final_compatibility_summary={"n":len(final_compatibility),"valid":sum(x["valid"] for x in final_compatibility),
         "invalid":sum(not x["valid"] for x in final_compatibility),"warning_cases":sum(bool(x["warnings"]) for x in final_compatibility),
         "error_case_ids":[x["case_id"] for x in final_compatibility if not x["valid"]],
         "warning_case_ids":[x["case_id"] for x in final_compatibility if x["warnings"]],
         "interpretation":"Read-only mechanical release-validator compatibility of final540 historical source-reviewed references; no field changes or semantic relabeling"}
    summary["final_reference_release_compatibility"]=final_compatibility_summary
    write_json(out/"summary.json",summary)
    for name,rows in [("model_comparison.csv",model_rows),("domain_comparison.csv",domain_rows),
                      ("model_domain_comparison.csv",cell_rows),("overall_comparison.csv",[overall]),
                      ("dimension_agreement.csv",dimensions)]:write_csv(out/name,rows)
    write_csv(out/"technical_unscored.csv",[{**r,"validation_errors":json.dumps(r["validation_errors"],ensure_ascii=False)} for r in technical])
    write_csv(out/"trajectory_transitions.csv",[{"automatic":k.split('->')[0],"reviewed":k.split('->')[1],"n":n} for k,n in sorted(transitions.items())])
    (out/"paired_trajectories.jsonl").write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs),encoding='utf-8')
    write_json(out/"release_validator_compatibility.json",{"summary":compatibility_summary,"cases":compatibility})
    write_json(out/"final_reference_release_compatibility.json",{"summary":final_compatibility_summary,"cases":final_compatibility})
    final_rows=[]
    for m in MODEL_ORDER:
        vals=[x["reviewed_states"] for x in pairs if x["model_alias"]==m]
        row=aggregate(vals)["trajectories"]
        final_rows.append({"model":m,"label":MODEL_NAMES[m],**row})
    write_csv(out/"reviewed_all540_by_model_SEPARATE.csv",final_rows)
    def percent(rate):return f'{rate*100:.1f}%'
    def line(row):
        return '| '+ ' | '.join([row['label'],str(row['paired_complete']),
                 f"{row['automatic_pass']}/{row['automatic_rate_denominator']}（{percent(row['automatic_pass_rate'])}）",
                 f"{row['reviewed_pass']}/{row['reviewed_rate_denominator']}（{percent(row['reviewed_pass_rate'])}）",
                 f"{row['review_minus_auto_percentage_points']:+.1f}",f"{row['verdict_agree']}/{row['paired_complete']}（{percent(row['agreement'])}）",
                 str(row['technical_unscored'])])+' |'
    text=['表D1 原始自动初评与源记录复核的同批对照',
          '| 被测模型 | 同批有效轨迹 | 自动通过率（通过/可判定） | 复核通过率（通过/可判定） | 复核−初评（百分点） | 整轨迹一致 | 格式失败另列 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    text += [line(x) for x in model_rows]
    text += [line(dict(overall,label='合计')),'',
        '注：两侧均使用527条原始输出完整的同一批轨迹；通过率分母为该侧通过与违反数之和，待定另计，因此两侧可判定分母不必相同。13条原始输出校验失败不归入通过或待定，仅在完整性统计中保留。复核来自助手/模型辅助源证据核查，不是独立人工金标准。',
        '',f"原始初评在540条中得到527条完整输出（{527/540*100:.1f}%），其余13条为技术未评分。完整样本中，初评通过率为{percent(overall['automatic_pass_rate'])}，复核为{percent(overall['reviewed_pass_rate'])}，差{overall['review_minus_auto_percentage_points']:.1f}个百分点；整轨迹判定459条一致，68条变化。",]
    order=lambda rows:' > '.join(x['name'] for x in rows)
    text += [f"同批模型总体顺序：初评为{order(rank_all['automatic'])}；复核为{order(rank_all['reviewed'])}。顺序{'相同' if same_ranking else '发生变化'}，不代表单条判定或分场景顺序都一致。"]
    for d in DOMAIN_ORDER:
        text.append(f"{DOMAIN_NAMES[d]}：初评{order(rank_scenes[d]['automatic'])}；复核{order(rank_scenes[d]['reviewed'])}。")
    text += ['', '68条变化包括初评通过→复核违反32条、初评通过→复核待定10条、初评违反→复核通过4条、初评违反→复核待定17条、初评待定→复核通过1条、初评待定→复核违反4条。通过数净减少37，不能只用一致率推断自动分数无偏。',
             'Qwen同批通过数94降至88，但待定由2增至15，可判定分母130降至117，故条件通过率反而上升；不能解释为复核提高了其业务表现。Llama总体通过数不变，但仍有2条逐例判定变化，不能把总比例相同当成每条都一致。',
             '最终540条的291通过/220违反/29待定属于完整参考集结果；同527对照中复核为288通过/211违反/28待定，两组分母不同，不可混算。',
             f"原首轮协议下完整527条，在当前发布版机械校验中{compatibility_summary['current_release_valid']}条有效、{compatibility_summary['current_release_invalid']}条另有校验差异，{compatibility_summary['current_release_warning_cases']}条带警告。此兼容性结果单列，不回改旧527分类，也不把它当语义准确率。",
             f"最终540条历史参考标注中，{final_compatibility_summary['valid']}条符合发布版严格校验，{final_compatibility_summary['invalid']}条有历史字段/引用约定差异，{final_compatibility_summary['warning_cases']}条带警告。差异含辅助严重度auxiliary/boundary、旧要求ID/revisions字段及关联到G2/G3/I2标注实体的链接终点；不等于源消息均不存在。完整参考聚合不依赖这些附加字段，不静默修改参考标签来满足新格式。",
             '', '用途：此表检查既有自动初评与源复核统计的差异；不调整检查器，不将同批一致率称为独立准确率。']
    (out/'附录用表与说明.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    after_hashes={p.name:sha(p) for p in auto_dir.glob('tr*.json')}
    assert before_hashes==after_hashes==locked
    integrity={"published_score_files_checked":540,"all_published_score_hashes_match_lock":True,
               "metadata_redaction": "raw_path only; original model answers and validity flags retained",
               "reference_sha256":sha(reference_path),"run_index_sha256":sha(index_path),**source_fingerprints,
               "score_hashes":before_hashes,"expected_agreement":"459/527","computed_agreement":f"{overall['verdict_agree']}/{overall['paired_complete']}",
               "original_invalid":13,"method":"trajectory_bench.grading.aggregate","new_api_calls":0,"new_business_trials":0}
    write_json(out/'integrity.json',integrity)
    source_info={"schema":"trajectory_process.original_auto_source.v1","original_score_count":540,
                 "original_score_hashes":portable["original_score_hashes"] if portable else locked,
                 "published_score_hashes":locked,
                 "manifest_cases":[{k:r[k] for k in ["case_id","domain","model_alias","task_id","repetition","input_sha256"]} for r in manifest.values()],
                 "api_config_safe":{"providers":{p:{"options":api_config["providers"][p]["options"]} for p,_ in settings}},
                 "original_source_fingerprints":source_fingerprints,
                 "note":"Frozen first automatic score hashes and minimum provenance; no keys or API request logs"}
    write_json(out/'comparison_sources.json',source_info)
    portable_usage='''便携重算（从benchmark目录执行，不调用API）
python -X utf8 analysis/score_comparison/compare_scores.py --raw-scores analysis_source/initial_scores540 --reference reference_data/reference/annotations.jsonl --run-index reference_data/run_index.json --out results/score_comparison

analysis_source/comparison_sources.json 与 initial_scores540 目录并列，保存540个原始输出的冻结hash及必要元数据。脚本在这些hash通过后计算原首轮协议下的527同批对照，并另外检查与当前发布版格式校验的兼容性。
reference_data/inputs 保留原评分输入，供机械引用校验，不调用模型。参考标签只用于本离线对照，不进入新轨迹的评分输入。
'''
    (out/'便携重算说明.txt').write_text(portable_usage,encoding='utf-8')
    print(json.dumps({"overall":overall,"models":[{k:r[k] for k in ['label','paired_complete','automatic_pass_rate','reviewed_pass_rate','review_minus_auto_percentage_points','agreement','technical_unscored']} for r in model_rows],"same_ranking":same_ranking},ensure_ascii=False))


if __name__=='__main__':main()

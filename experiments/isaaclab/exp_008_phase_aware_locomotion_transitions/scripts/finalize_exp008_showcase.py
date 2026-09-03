"""Write video/social manifests after the real closeout recording."""
import hashlib, json, subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[4]; M=R/"media/exp_008_closeout"; O=R/"results/exp_008_phase_aware_locomotion_transitions/final_closeout"
def probe(p):
 d=json.loads(subprocess.check_output(["ffprobe","-v","quiet","-print_format","json","-show_streams","-show_format",str(p)],text=True));s=d["streams"][0]
 return {"path":str(p.relative_to(R)).replace("\\","/"),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size,"duration_seconds":float(d["format"]["duration"]),"width":s["width"],"height":s["height"],"fps":eval(s["avg_frame_rate"])}
names=["scene1_stand_walk_stand.mp4","scene2_walk_to_run_2p6.mp4","scene3_walk_to_run_2p8.mp4","exp008_g1_state_graph_closeout_showcase.mp4"]
tele=[json.loads((M/n).read_text()) for n in ["StandWalkStand_telemetry.json","WalkToRun26_telemetry.json","WalkToRun28_telemetry.json"]]
manifest={"recording_status":"COMPLETED_WITH_RECORDED_SCENE_FAILURE","showcase":"EXP_007_FORMAL_CAPABILITIES_REPLAYED","files":[probe(M/n) for n in names],"scenes":tele,"all_three_scenes_recorded":True,"robot_visible":True,"visual_validation_frame":"media/exp_008_closeout/validation_frame.png","unsupported_transition_executed":sum(x["unsupported_transition_executed"] for x in tele),"routing_errors":sum(x["routing_error"] for x in tele),"note":"2.8 m/s reached RUN_LOW but failed the fixed-seed 7-second hold diagnostic; no seed selection was performed."}
(O/"video_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
social={"linkedin":"research/exp_008_linkedin_post_ja.md","x":"research/exp_008_x_post_ja.md","video":"media/exp_008_closeout/exp008_g1_state_graph_closeout_showcase.mp4","validation":{"isaac_sim_disclosed":True,"unsupported_capability_not_claimed":True,"run_to_walk_failure_disclosed":True,"exp008_new_capability_claimed":False,"go2_is_project_decision":True,"github_url_included":False}}
(O/"social_copy_manifest.json").write_text(json.dumps(social,indent=2)+"\n")
p=O/"exp008_closeout.json";d=json.loads(p.read_text());d["commit_hashes"]={"closeout":"2ba9510da2cb1ab557953c610072ca6c5b3bff15","showcase":"THIS_COMMIT"};d["ending_head"]="SHOWCASE_COMMIT_CONTAINING_THIS_MANIFEST";p.write_text(json.dumps(d,indent=2)+"\n")

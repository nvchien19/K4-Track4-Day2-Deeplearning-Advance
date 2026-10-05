import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'submissions/2A202602926_NguyenVanChien/code'
train_path = CODE / 'train.py'
train = train_path.read_text(encoding='utf-8')
train = train.replace('torch.save({"epoch": best_ep, "state": best_state}, rdir / "best.pt")',
                      'torch.save({"epoch": best_ep, "state": best_state, "cfg": dataclasses.asdict(cfg)}, rdir / "best.pt")')
train = train.replace('        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)',
                      '        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch+1}/{cfg.epochs} "\n'
                      '              f"loss={tr[\'train_loss\']:.4f} val_f1={m[\'macro_f1\']:.4f} "\n'
                      '              f"train_s={tr[\'time_s\']:.1f}", flush=True)\n'
                      '        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)')
ast.parse(train)
train_path.write_text(train, encoding='utf-8')
nb = json.loads((CODE / 'colab_full.ipynb').read_text(encoding='utf-8'))
payload = {'eval.py': (ROOT / 'eval.py').read_text(encoding='utf-8')}
payload.update({'code/' + p.name: p.read_text(encoding='utf-8') for p in CODE.glob('*.py')})
bootstrap = '''from pathlib import Path
import json, os, sys
os.chdir('/content')
PAYLOAD = json.loads(PAYLOAD_JSON)
for name, source in PAYLOAD.items():
    dest = Path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source, encoding='utf-8')
for folder in ('curves', 'predictions', 'runs', 'eval_out'):
    Path(folder).mkdir(exist_ok=True)
print('Embedded code and eval.py ready')
'''.replace('PAYLOAD_JSON', repr(json.dumps(payload, ensure_ascii=False)))
nb['cells'].insert(2, {'cell_type': 'code', 'metadata': {}, 'source': bootstrap.splitlines(True), 'execution_count': None, 'outputs': []})
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code':
        if 'drive.mount(' in source:
            source = "print('Automatic Zenodo download; no Drive mount required')\n"
        source = source.replace('FULL = False', 'FULL = True').replace('batch_size=64', 'batch_size=32')
        source = source.replace('BEST_BB = "resnet50"  # DOI theo ket qua Buoc 1 (F1 cao nhat / can bang voi do tre)',
                                'BEST_BB = df1.sort_values("val_macro_f1", ascending=False).iloc[0]["backbone"]\nprint("Selected backbone on validation:", BEST_BB)')
        source = source.replace('EXP = "T00"  # DOI thanh exp tot nhat Buoc 2',
                                'EXP = max(["T00"] + [eid for eid, _ in EXPS], key=lambda eid: json.load(open("runs/"+eid+"/seed0/summary.json"))["val_macro_f1"])\nprint("Selected training recipe on validation:", EXP)')
        if 'FINAL_CFG = dict(' in source:
            source = '''import dataclasses, gc, json, torch, numpy as np
import dataset as D, model as M
from train import Config, run
from inference import predict_logits, fit_temperature, apply_temperature
from eval import save_predictions
selected = json.load(open('runs/'+EXP+'/seed0/config.json'))
FINAL_CFG = {k: v for k, v in selected.items() if k not in ('exp_id', 'seed', 'save_test_predictions')}
FINAL_CFG.update(epochs=12, save_test_predictions=False)
BASE_FINAL = dict(backbone=BEST_BB, epochs=12, batch_size=32, images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, save_test_predictions=False)
json.dump({'selected_experiment': EXP, 'final': FINAL_CFG, 'baseline': BASE_FINAL}, open('final_selection.json', 'w'), indent=2)
device = torch.device('cuda')
tr_df, va_df, te_df = D.load_split(LABELS_DIR, 0)
valloader = D.make_loader(va_df, IMAGES_DIR, D.build_transforms(False, 224), 32, False, None, 2)
testloader = D.make_loader(te_df, IMAGES_DIR, D.build_transforms(False, 224), 32, False, None, 2)
calibration = {}
for exp, recipe in [('F01', FINAL_CFG), ('BASE12', BASE_FINAL)]:
    for seed in (0, 1, 2):
        run(Config(exp_id=exp, seed=seed, **recipe))
        ck = torch.load(f'runs/{exp}/seed{seed}/best.pt', map_location='cpu', weights_only=True)
        c = ck['cfg']
        n = M.build_model(c['backbone'], False, 9, c.get('drop_rate', 0.0), 'finetune')
        n.load_state_dict(ck['state']); del ck
        n.to(device).eval()
        fv, yv, lv = predict_logits(n, valloader, device)
        temperature = fit_temperature(lv, yv) if exp == 'F01' else 1.0
        calibration[f'{exp}_seed{seed}'] = temperature
        save_predictions(f'predictions/{exp}_seed{seed}_val.csv', fv, yv, apply_temperature(lv, temperature))
        # Exactly one test forward pass per model; derive both output versions from the same logits.
        ft, yt, lt = predict_logits(n, testloader, device)
        save_predictions(f'predictions/{exp}uncal_seed{seed}_test.csv', ft, yt, apply_temperature(lt, 1.0))
        save_predictions(f'predictions/{exp}_seed{seed}_test.csv', ft, yt, apply_temperature(lt, temperature))
        print(exp, 'seed', seed, 'test completed; temperature', temperature, flush=True)
        del n
        gc.collect(); torch.cuda.empty_cache()
json.dump(calibration, open('final_temperatures.json', 'w'), indent=2)
'''
        if source.startswith('!python eval.py score'):
            source = source.replace('predictions/T00_seed', 'predictions/BASE12_seed').replace('--tag T00', '--tag BASE12')
        if 'shutil.make_archive("colab_outputs"' in source:
            source = '''import json, glob, pandas as pd, zipfile
from pathlib import Path
rows = []
for s in sorted(glob.glob('runs/*/seed*/summary.json')):
    d = json.load(open(s)); d['path'] = s
    d.update(json.load(open(Path(s).parent / 'config.json')))
    rows.append(d)
pd.DataFrame(rows).to_csv('all_runs_summary.csv', index=False)
with pd.ExcelWriter('colab_results.xlsx') as writer:
    pd.DataFrame(rows).to_excel(writer, sheet_name='Runs', index=False)
    df1.to_excel(writer, sheet_name='Backbones', index=False)
    if Path('inference_results.json').exists():
        pd.DataFrame(json.load(open('inference_results.json'))['latency']).T.to_excel(writer, sheet_name='Latency')
with zipfile.ZipFile('colab_outputs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('code', 'predictions', 'curves', 'eval_out', 'runs'):
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.json', '.csv', '.png'):
                z.write(p, str(p))
    for pattern in ('*.csv', '*.json', '*.xlsx', 'eval.py'):
        for p in Path('.').glob(pattern): z.write(p, str(p))
print('DONE: colab_outputs.zip includes code, predictions, curves, evaluation, summaries and workbook', flush=True)
from google.colab import files
files.download('colab_outputs.zip')
'''
        if 'from train import Config, run\nBACKBONES' in source:
            source = source.replace('from train import Config, run', '''from train import Config, run as original_run
import gc, json, dataclasses
def run(cfg):
    folder = Path(cfg.out_dir) / cfg.exp_id / f'seed{cfg.seed}'
    if (folder/'summary.json').exists() and (folder/'config.json').exists():
        previous = json.load(open(folder/'config.json'))
        if previous == dataclasses.asdict(cfg):
            print('Reuse completed:', cfg.exp_id, cfg.seed, flush=True)
            return json.load(open(folder/'summary.json'))
    gc.collect(); torch.cuda.empty_cache()
    return original_run(cfg)
del net, opt, x, y, xb, yb
gc.collect(); torch.cuda.empty_cache()''')
        # Keep wrapper active in final cell.
        source = source.replace('from train import Config, run\n', 'from train import Config\n') if 'FINAL_CFG =' in source else source
        source = source.replace('print("GPU:",', 'assert torch.cuda.is_available(), "GPU runtime required"\nprint("GPU:",')
        if '%pip' not in source and not source.startswith('!'):
            ast.parse(source)
        cell['execution_count'] = None
        cell['outputs'] = []
    cell['source'] = source.splitlines(True)
nb['cells'][0]['source'] = ['# DeepWeeds — full automated Colab run\n', 'Embedded code; GPU required; 12 epochs; selection uses validation only; test once per final model.\n']
nb['metadata']['accelerator'] = 'GPU'
nb['metadata']['colab'] = {'gpuType': 'T4'}
target = CODE / 'colab_autorun.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target)
print('Notebook verified: embedded code + parsed Python cells')

import json, re
raw = open('/Volumes/500G/models/mlx-community/Qwen3.5-0.8B-bf16/config.json').read()
for key in ['rope_theta', 'partial_rotary_factor']:
    for m in re.finditer('"' + key + '"', raw):
        s = max(0, m.start() - 120)
        e = min(len(raw), m.end() + 5)
        print('===', key, '===')
        print(raw[s:e].replace('\n', ' '))
        print()

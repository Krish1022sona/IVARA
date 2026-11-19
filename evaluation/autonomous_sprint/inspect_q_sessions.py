import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open('evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json', 'r', encoding='utf-8') as f:
    sample_list = json.load(f)

inst = next(i for i in sample_list if i['question_id'] == '10e09553')
print('Question:', inst['question'])
print('Gold Answer:', inst['answer'])
print('Answer session IDs:', inst['answer_session_ids'])
for sid, s in zip(inst['haystack_session_ids'], inst['haystack_sessions']):
    if sid in inst['answer_session_ids']:
        print('=== TARGET SESSION ===', sid)
        for turn in s:
            txt = turn.get('content', turn.get('text', ''))
            print(' ', turn.get('role', turn.get('speaker')), ':', repr(txt[:150]))

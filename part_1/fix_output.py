import json

fixed_output = []

with open('output.json', 'r') as _f:
    data = json.load(_f)
    for item in data:
        _id = item['id']
        snip = item['snippet']
        for s in snip:
            if type(s) != str:
                print('Removing', s, 'from', snip, 'in', _id)
                snip.remove(s)
        fixed_output.append({
            'id': _id,
            'snippet': snip
        })
        
print('output fixed')
print(len(fixed_output))
with open('output_fix_1.json', 'w') as _o:
    json.dump(fixed_output, _o, indent=None)

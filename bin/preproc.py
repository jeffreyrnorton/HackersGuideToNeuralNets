# Preprocessor of a markdown file to insert files into a markdown.

import sys

if len(sys.argv) < 2:
    print('Usage: {} <filename>.md'.format(sys.argv[0]))

with open(sys.argv[1]) as f:
    content = f.readlines()
    content = [x.rstrip('\n') for x in content] 
    for l in content:
        b, e = l.find("<<")+2, l.find(">>")
        if b >= 0 and e >= 0:
            ifname = l[b:e]
            with open(ifname) as ff:
                icont = ff.readlines()
                icont = [x.rstrip('\n') for x in icont] 
                for ll in icont:
                    print(ll)
        else:
            print(l)

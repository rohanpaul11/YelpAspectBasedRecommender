with open('test.qrels', 'r') as f:
	with open('modtest.qrels', 'w') as f1:
		fmap = {}
		for line in f.readlines():
			arr = line.split()
			if arr[0] in fmap:
				# print("in map " + arr[0])
				if arr[2] in fmap[arr[0]]:
					print("duplicate " + line)
				else:
					fmap[arr[0]].add(arr[2])
					# print("non duplicate " + line)
					f1.write(line)
			else:
				# print("adding " + arr[0])
				fmap[arr[0]] = set([arr[2]])
				f1.write(line)
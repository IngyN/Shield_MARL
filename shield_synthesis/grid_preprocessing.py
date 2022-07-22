# input = map_name resolution (width, length)
import numpy as np
import sys

def preprocessing (map, res = [3,3]):
	
	# Read map from map_name to get width and length
	width = len(map)
	length = len(map[0])
	# print(" dim: ", width, length)

	lswitch = 0
	wswitch = 0
	

	while length%res[1] or width%res[0]:

		if length%res[1] != 0:
			length += 1
			if lswitch == 0:
				# pad first column
				for i in range(width ):
					map[i].insert(0,1)
				# map.insert(0, paddingh)
				lswitch = 1
			else:
				# pad last column
				for i in range(width):
					map[i].insert(length,1)
				lswitch = 0
				
		paddingh = [1]*length

		if width%res[0] != 0:
			width += 1
			if wswitch == 0:
				# pad first row
				map.insert(0, paddingh)
				wswitch = 1
			else:
				# pad last row
				map.insert(len(map), paddingh)
				wswitch = 0
 
		# print(" dim: ", width, length)

	#print(" dim padded: ", width, length)
	#print(map)
	# fill smap with shieldnum for each cell
	smap = np.zeros([width, length])
	for i in range(width):
		for j in range(length):
			# 1 based shield number
			shieldnum = int((i/res[0]))*int(length/(res[1]))+ int(j/res[0]) + 1
			smap[i][j] = shieldnum

	#print("SMAP: \n", smap)
	# Create coordinates list for each shield
	# Create vector1 = [0,1,2,..., (length/res[1]-1)]*res[1]
	# Create vector2 = [0,1,2,..., (width/res[0]-1)]*res[0]
	# Coordinates = [(vector1[i], vector2[i]) for i in range(length/res[1], width/res[0]) ]
	'''TODO : iron out exacty how to choose width, length ''' 
	v1 = [i for i in range (0, int(width/res[0]))]
	v1 = np.multiply(v1, res[0])

	v2 = [i for i in range (0, int(length/res[1]))]
	v2 = np.multiply(v2, res[1])

	coord = [[v1[i], v2[j]] for i in range(int(width/res[0])) for j in range(int(length/res[1]))]
	#print(" Coord: ", coord)

	return map, smap, coord

# main
if __name__ == "__main__":

	wr = None
	lr = None

	map_name = sys.argv[1]
	with open('maps/'+map_name+".txt") as textFile:
		map = [line.split() for line in textFile]
	map = [[int(e) for e in row] for row in map ]
	print(map)

	if len(sys.argv) > 2:
		wr= int(sys.argv[2])
		lr= int(sys.argv[3])

	if wr == None or lr == None :
		preprocessing(map)
	else:
		preprocessing(map,[wr,lr])



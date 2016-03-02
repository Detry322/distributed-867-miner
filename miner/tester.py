import subprocess

MINER = "./miner"
proc = None
miner_in = None
miner_out = None
triples = {}

TEST_CONFIG="H 56a95696f0520b7100eacc3fea9418e38e72de35c6999bbf162a0b516cbd3a4f 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824 0000000000000032 00052d0e025dd563 00"

def read_response():
  global miner_in, miner_out, triples
  line = miner_out.readline().strip()
  while True:
    if not line:
      pass
    if line.startswith("="):
      print "DEBUG: " + line
    if line.startswith("A "):
      a, ts, s, d, l = line.split(" ")
      print "Triple: ", (s, d, l)
      if d not in triples:
        triples[d] = [(int(s),int(l))]
      else:
        triples[d].append((int(s),int(l)))
    line = miner_out.readline().strip()

def start_test():
  global proc, miner_in, miner_out
  proc = subprocess.Popen([MINER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  miner_in = proc.stdin
  miner_out = proc.stdout
  miner_in.write(TEST_CONFIG + "\n")
  miner_in.write("A\n")
  read_response()


if __name__ == "__main__":
  start_test()

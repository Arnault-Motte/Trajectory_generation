
import sys  # noqa: I001
import os

current_path = os.getcwd()
sys.path.append(
    os.path.abspath(current_path)
)

print("Current working directory:", current_path)
print(os.path.dirname(__file__))

import minisky

minisky.init()

test = "DEFWPT WPT_1 48.71088333 2.0090556"
rest = "CRE 17LQ_000, A320, 48.15824347027277, 1.486323429987981, 34.02884128397152, 1000, 375.0"
last = "ADDWPT 17LQ_000 WPT_1"
minisky.sim.reset()
# minisky.traf.cre('KL315', lat=52.0, lon=4.0, hdg=45, alt=5000, spd=250)
# minisky.stack.stack('KL315 ADDWPT HELEN FL100 250')

minisky.stack.stack(test)
minisky.stack.stack("DEFWPT WPT_2 52.71088333 1.0090556")
minisky.stack.stack(rest)
minisky.stack.stack("CRE 17LQ_001, A320, 48.15824347027277, 1.486323429987981, 34.02884128397152, 1000, 375.0")
minisky.stack.stack(last)
minisky.stack.stack("ADDWPT 17LQ_000 WPT_2")
print(minisky.navdb.wpid.count('WPT_1'))

minisky.sim.simdt = 10

for i in range(60):
    print(minisky.navdb.wpid.count('WPT_1'))
    minisky.sim.step()
    print(f"time-{minisky.sim.simt}s, positions: {minisky.traf.lat} {minisky.traf.lon}")
    print(type(minisky.traf.lat))
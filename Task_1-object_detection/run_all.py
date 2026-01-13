import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"running {script_name}...")
    print('='*50)
    result = subprocess.run([sys.executable, script_name], cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def main():
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./outputs/detections', exist_ok=True)

    print("step 1: training model...")
    if not run_script('train.py'):
        print("training failed!")
        return

    print("\nstep 2: evaluating model...")
    if not run_script('evaluate.py'):
        print("evaluation failed!")
        return

    print("\nstep 3: generating visualizations...")
    if not run_script('visualize.py'):
        print("visualization failed!")
        return

    print("\n" + "="*50)
    print("all done!")
    print("="*50)

if __name__ == '__main__':
    main()

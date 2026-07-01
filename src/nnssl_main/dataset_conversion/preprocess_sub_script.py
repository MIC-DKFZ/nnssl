def main():
    total_parts = 600
    dataset = 801
    np = 8
    for i in range(total_parts):
        bsub_command = f"""bsub -n {np} -R "rusage[mem=120G]" -q verylong -L /bin/bash "source /home/t006d/nnsslrc && nnssl_preprocess -d {dataset} -np {np} -part {i} -total_parts {total_parts} -c 3d_fullres -plans_name nnsslPlans" """
        print(bsub_command)
        if (i + 1) % 10 == 0:
            print()
    pass


if __name__ == "__main__":
    main()

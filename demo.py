from flowbench.evals import evaluate

if __name__ == '__main__':
    model, results = evaluate(
        model_name='RAFT',
        dataset='Sintel-Final',
        retrieve_existing=False,
        threat_model='3DCommonCorruption',
        severity=3,
    )
    print(results)

    model, results = evaluate(
        model_name='RAFT',
        dataset='KITTI2015',
        retrieve_existing=True,
        threat_model='PGD',
        iterations=20, epsilon=8/255, alpha=0.01,
        lp_norm='Linf', optim_wrt='ground_truth',
        targeted=True, target='zero',
    )

    model, results = evaluate(
        model_name='RAFT',
        dataset='Sintel-Final',
        retrieve_existing=False,
        threat_model='Adversarial_Weather',
        weather='snow', num_particles=10000, targeted=True, target='zero',
    )
    print(results)

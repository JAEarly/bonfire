import optuna.visualization as viz
from optuna.trial import TrialState

from bonfire.tune.tune_util import load_study, generate_figure
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('study_uid')
    parser.add_argument('study_dir')
    parser.add_argument('direction', choices=['maximize', 'minimize'])
    args = parser.parse_args()
    return args.study_uid, args.study_dir, args.direction


def run():
    study_uid, study_dir, direction = parse_args()
    study = load_study(study_uid, study_dir, direction=direction)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    early_stopped_trials = [t for t in study.trials if
                            'early_stopped' in t.user_attrs and t.user_attrs['early_stopped']]
    all_trial_scores = [t.value for t in study.trials if t.value is not None]
    complete_trial_scores = [t.value for t in study.trials if t.state == TrialState.COMPLETE]

    top_scores = np.argsort(all_trial_scores)
    if direction == 'maximize':
        top_scores = top_scores[::-1]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Number of early stopped trials: ", len(early_stopped_trials))

    print("Trial statistics:")
    print("  Max: {:.3f}".format(max(complete_trial_scores)))
    print("  Avg: {:.3f}".format(np.mean(complete_trial_scores)))
    print("  Min: {:.3f}".format(min(complete_trial_scores)))

    print("Top by key metric:")
    mean_accs = {}
    for trial_id, trial in enumerate(study.trials):
        if 'test_accuracies' in trial.user_attrs:
            mean_acc = np.mean(trial.user_attrs['test_accuracies'])
            mean_accs[trial_id] = mean_acc
    if mean_accs:
        sorted_accs = sorted(mean_accs.items(), key=lambda x: x[1], reverse=True)
        for i in range(5):
            print("  {:d}: {:.3f} ({:d})".format(i+1, sorted_accs[i][1], sorted_accs[i][0]))

    print("Top 3:")
    for i in range(3):
        print("  {:d}: {:.3f} ({:d})".format(i+1, all_trial_scores[top_scores[i]], top_scores[i]))

    auto_open = True
    generate_figure(viz.plot_optimization_history, study_dir, study, auto_open)
    generate_figure(viz.plot_intermediate_values, study_dir, study, auto_open)
    generate_figure(viz.plot_slice, study_dir, study, auto_open)
    generate_figure(viz.plot_param_importances, study_dir, study, auto_open)
    generate_figure(viz.plot_param_importances, study_dir, study, auto_open,
                    target=lambda t: t.duration.total_seconds(), target_name="duration")

    while True:
        val = input("Query: ")
        if val == 'q':
            break
        idx = int(val)
        trial = study.trials[idx]

        print("  Value: ", trial.value)
        if 'test_accuracy' in trial.user_attrs:
            print("  Acc: {:.3f}".format(trial.user_attrs['test_accuracy']))
        else:
            print(trial.user_attrs)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run()

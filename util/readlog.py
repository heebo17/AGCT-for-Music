import matplotlib.pyplot


def plot(filename: str, title: str="Training accuracy"):
    x = []
    accuracy = []
    tf_ratio = []
    epoch_avg = []
    eval_acc = []
    stride = None
    with open(filename) as f:
        ntrain = 0
        for line in f:
            if line.startswith(" Batch"):
                # print_every summary statistics
                word = line.split()
                if stride is None:
                    assert(word[1].startswith("0-"))
                    stride = int(word[1][2:].strip(",.:()/"))
                # loss = float(word[3].strip(",.:()/"))
                acc = float(word[5].strip(",.:()/"))
                # acc_notf = float(word[6].strip(",.:()/"))
                # acc_tf = float(word[7].strip(",.:()/"))
                tf = float(word[10].strip(",.:()/"))
                ntrain += stride
                x.append(ntrain)
                accuracy.append(acc)
                tf_ratio.append(tf)
            elif line.startswith("Epoch finished"):
                # per epoch statistics
                word = line.split()
                nbatch = int(word[2].strip(",.:()/"))  # batches per epoch
                # loss_avg = float(word[5].strip(",.:()/"))
                acc_avg = float(word[7].strip(",.:()/"))
                # acc_avg_notf = float(word[8].strip(",.:()/"))
                # acc_avg_tf = float(word[9].strip(",.:()/"))
                if stride is None:
                    # TODO: deduce average tf_ratio from accuracy distribution
                    #       (actually, tf_ratio = (acc_avg-acc_avg_notf) /
                    #                             (acc_avg_tf-acc_avg_notf),
                    #        but be careful when plotting (plot what?))
                    raise NotImplementedError("Need to see at least one "
                                              "print_every summary.")
                for i in range(stride, nbatch-1, stride):
                    epoch_avg.append(acc_avg)
                ntrain += nbatch-i
            elif line.startswith("Completed evaluation"):
                word = line.split()
                # eloss = float(word[3].strip(",.:()/"))
                eacc = float(word[5].strip(",.:()/"))
                eval_acc.append(eacc)

    matplotlib.pyplot.plot(x, accuracy, label='training accuracy')
    matplotlib.pyplot.plot(x, epoch_avg, linestyle="--",
                           label='training accuracy (epoch average)')
    matplotlib.pyplot.plot(x, tf_ratio, linestyle="dotted",
                           label='treacher forcing ratio')
    if len(eval_acc) is not 0:  # maybe they turned eval off
        assert(ntrain == nbatch*len(eval_acc))
        eval_at = [i+nbatch for i in range(0, ntrain, nbatch)]
        matplotlib.pyplot.scatter(eval_at, eval_acc, marker=".", c="red",
                                  label="evaluation accuracy")
    matplotlib.pyplot.xlabel("batch number")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

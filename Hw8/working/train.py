import time
import torch.utils.data as data
import torch
from utils import schedule_sampling, tokens2sentence, save_model, EN2CNDataset, computebleu, infinit_iter, build_model, \
    Configuration
import numpy as np
import torch.nn as nn


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps):
    '''
    return model, optimizer, losses
    '''
    # set train in train model
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling())
        # targets 的第一个 toekn 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # why ???? 处理梯度爆炸
        optimizer.step()

        loss_sum += loss.item()

        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print('\r', 'train [{}] loss {:.3f}, Perplexity {:.3f}    '.format(
                total_steps + step + 1, loss_sum, np.exp(loss_sum)
            ), end=' ')
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def test(model, dataloader, loss_function):
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []

    for sources, targets in dataloader:
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一个 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        # 将预测结果转为文字
        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
        targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 计算 Bleu Score
        bleu_score += computebleu(preds, targets)

        n += batch_size

    return loss_sum / len(dataloader), bleu_score / n, result


def train_process(config):
    start = time.time()
    # 准备训练资料
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinit_iter(train_loader)
    # 准备检验资料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    # 构建模型
    model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)  # ？？？ 是 ignore bias 的 grad 吗

    # 训练过程
    train_loss, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while total_steps < config.num_steps:
        # 训练模型
        model, optimizer, losses = train(model, optimizer, train_iter, loss_function, total_steps,
                                         config.summary_steps)
        train_loss += losses
        # 检验模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += config.summary_steps
        print('\r', 'val [{}] loss {:.3f}, Perplexity: {:.3f}, bleu score: {:.3f}, used {} seconds      '.format(
            total_steps, val_loss, np.exp(val_loss), bleu_score, int(time.time() - start)
        ))

        # 储存模型和结果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for l in result:
                    print(l, file=f)

    return train_loss, val_losses, bleu_scores


def test_process(config):
    # 准备测试资料
    test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    # 建构模型
    model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print('Finish build model')
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 测试模型
    test_loss, bleu_score, result = test(model, test_loader, loss_function)
    # 储存结果
    with open(f'./test_output.txt', 'w') as f:
        for line in result:
            print(line, file=f)

    return test_loss, bleu_score

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Configuration()
    print('config:\n',vars(config),'\n')
    train_losses, val_losses, bleu_scores = train_process(config)

    # 在执行 test 之前 需要在 config 中设定要载入的模型位置
    config.load_model = True
    print('config:\n',vars(config),'\n')
    test_loss, bleu_score = test_process(config)
    print(f'test loss: {test_loss}, bleu_score: {bleu_score}')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('次数')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()

    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('次数')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.show()

    plt.figure()
    plt.plot(bleu_scores)
    plt.xlabel('次数')
    plt.ylabel('BLEU score')
    plt.title('BLEU score')
    plt.show()
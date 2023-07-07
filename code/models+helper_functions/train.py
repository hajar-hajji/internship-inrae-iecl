from datetime import datetime, timedelta

# fonction pour formater le temps écoulé (exprimé en sec) pour la mise en forme de la verbose (fonction train)
def format_time(seconds):
  
    td = timedelta(seconds=seconds)
    d = datetime(1, 1, 1) + td

    if d.hour > 0:
        return f"{d.hour}h {d.minute:02d}min {d.second:02d}s"
    elif d.minute > 0 :
        return f"{d.minute:02d}min {d.second:02d}s"
    elif d.second > 0:
      return f"{d.second:02d}s {d.microsecond // 1000}ms"
    else:
      return f"{d.microsecond // 1000}ms"

def test(mod, my_loss_fct):
    mod.train(False)
    total_loss, nbatch = 0., 0
    for batch in test_loader:
        sequences, target = batch
        pred_target = mod(sequences)
        loss = my_loss_fct(target, pred_target)
        total_loss += loss.item()
        nbatch += 1
    total_loss /= float(nbatch)
    mod.train(True)
    return total_loss

def train(mod, nepochs, learning_rate, my_loss_fct, verbose=True):

    print('Start training...')

    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    test_loss_vect = np.zeros(nepochs)
    train_loss_vect = np.zeros(nepochs)

    start_time_global = time.time()

    for epoch in range(nepochs):
        mod.reset_hidden_state()
        test_loss = test(mod, my_loss_fct)
        total_loss, nbatch = 0., 0

        start_time = time.time()

        for batch in train_loader:
            sequences, target = batch
            optim.zero_grad()
            pred_target = mod(sequences)
            loss = my_loss_fct(target,pred_target)  # fonction de loss personnalisée
            total_loss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()

        total_loss /= float(nbatch)
        test_loss_vect[epoch] = test_loss
        train_loss_vect[epoch] = total_loss

        end_time = time.time()

        if verbose and (epoch+1)%10==0:
          epoch_time = end_time - start_time
          print(f"Epoch [{epoch+1}/{nepochs}] [==============================] {format_time(epoch_time)}/step - train loss: {total_loss:.4f} - validation loss: {test_loss:.4f}")

    end_time_global = time.time()  
    global_time = end_time_global - start_time_global  

    if verbose:
      print()
      print('End training...')
      print(f"Fin Epoch [{epoch+1}/{nepochs}] [==============================] training time: {format_time(global_time)} - train loss: {total_loss:.4f} - validation loss: {test_loss:.4f}")

    return train_loss_vect, test_loss_vect

# torch.save(mod, path)


import bz2
# this is the size of our encoded representations
encoding_dim = 300  # 300 floats -> compression of factor ~20, assuming the input is ~600k bools

with bz2.BZ2File("training_data0.pbz", "rb") as fin:
    fireplace.cards.db.initialize()
    training_set = []
    while True:
        try:
            sim1 = HSsimulation()
            while True:
                actions = sim1.actions()
                if len(actions) == 1:
                    # end turn
                    fireplace.utils.play_turn(sim1.game)
                    # play opponent turn
                    fireplace.utils.play_turn(sim1.game)
                else:
                    choosen_action = random.choice(actions)
                    observation, action, next_observation, terminal = sim1.step(choosen_action)
                    if choosen_action is not None:
                        training_tuple = (observation, action.bow, next_observation)
                        training_set.append(training_tuple)
        except GameOver as e:
            games_finished += 1
            # if games_finished > 10:
            #     break
            nr_tuples += len(training_set)
            print(games_finished, nr_tuples)
            pickle.dump(training_set, fout)
            fout.flush()
            training_set = []
        except TypeError as e:
            print("game failed")
        except Exception as e:
            print(str(e))
            break

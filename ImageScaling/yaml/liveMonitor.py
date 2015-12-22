from pylearn2.train_extensions.live_monitoring import LiveMonitor
lm = LiveMonitor()
lm.follow_channels(['train_y_mse', 'valid_y_mse'])

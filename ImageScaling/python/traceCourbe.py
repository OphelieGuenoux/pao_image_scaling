from pylearn2.train_extensions.live_monitoring import LiveMonitor

lm = LiveMonitor(address="127.0.0.1", req_port=5655)
lm.follow_channels(['app_objective', 'val_objective'])

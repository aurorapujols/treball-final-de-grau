import joblib
import torch


if __name__ == '__main__':
    main()

    from models.ssl_model import SSLResNet
    ssl_model = SSLResNet(res_net_dim=512, projection_dim=256)
    state = torch.load("../../../data/upftfg26/apujols/models/ssl_final_model_1.0.pt", map_location='cpu', weights_only=False)
    ssl_model.load_state_dict(state)
    torch.save(ssl_model.state_dict(), 'logs/encoder.pt')

    clf = joblib.load("../../../data/upftfg26/apujols/models/mlp_model_1.0.pt")
    torch.save(clf.state_dict(), 'logs/classifier.pt')
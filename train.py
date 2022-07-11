def init_model(opt):

    if opt.grayscale:
        out_features = 1
    else:
        out_features = 1

    if opt.model == 'mlp':

        if opt.multiscale:
            m = modules.MultiscaleCoordinateNet
        else:
            m = modules.CoordinateNet

        model = m(nl=opt.activation,
                  in_features=2,
                  out_features=out_features,
                  hidden_features=opt.hidden_features,
                  num_hidden_layers=opt.hidden_layers,
                  w0=opt.w0,
                  pe_scale=opt.pe_scale,
                  no_pe=opt.no_pe,
                  integrated_pe=opt.ipe)
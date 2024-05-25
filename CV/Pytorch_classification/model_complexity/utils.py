
def conv2d_cx(cx, in_c, out_c, k, *, stride=1, groups=1, bias=False, trainable=True):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, c = cx["h"], cx["w"], cx["c"]
    assert c == in_c
    h, w = (h-1) // stride+1, (w-1) // stride + 1
    cx["h"] = h
    cx["w"] = w
    cx["c"] = out_c
    cx["flops"] += k*k*in_c*out_c*h*w //groups + (out_c if bias else 0)
    cx["params"] += k*k*in_c*out_c // groups + (out_c if bias else 0)
    cx["acts"] += out_c*h*w
    if trainable is False:
        cx["freeze"] += k*k*in_c*out_c // groups + (out_c if bias else 0)
    return cx


def pool2d_cx(cx, in_c, k, *, stride=1):
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, c = cx["h"], cx["w"], cx["c"]
    assert c == in_c
    h, w = (h - 1) // stride+1, (w-1) // stride + 1
    cx["h"] = h
    cx["w"] = w
    cx["c"] = c
    return cx

def 
# -------------------------
# Utility: Unflatten Action
# -------------------------
def unflatten_action(index: int, shape):
    total = int(np.prod(shape))
    one_hot = jnp.zeros(total)
    one_hot = one_hot.at[index].set(1.0)
    return one_hot.reshape(shape)
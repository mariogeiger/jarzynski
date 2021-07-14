import jax.numpy as jnp
import jax


def ball_ball_collision_time(a, b):
    x = b['x'] - a['x']
    v = b['v'] - a['v']
    r = a['r'] + b['r']
    xv = jnp.sum(x * v)

    def f(pa):
        x, v, r = pa
        vv = jnp.sum(v * v)
        dis = xv * xv - vv * (jnp.sum(x * x) - r * r)

        def g(pa):
            xv, vv, dis = pa
            return (-xv - jnp.sqrt(dis)) / vv

        return jax.lax.cond(dis > 0.0, g, lambda _: jnp.inf, (xv, vv, dis))

    return jax.lax.cond(xv < 0.0, f, lambda _: jnp.inf, (x, v, r))


def ball_wall_collision_time(a, w):
    x = a['x'] - w['x']
    v = a['v'] - w['v']
    n = jnp.cross(w['j'], w['k'])
    njck = jnp.linalg.norm(n)
    n = n / njck
    vn = jnp.sum(v * n)
    xn = jnp.sum(x * n)

    def f(pa):
        x, v, n, xn, vn, r, njck, j, k = pa
        t = -xn / vn - jnp.abs(r / vn)
        x = x + t * v
        xcn = jnp.cross(x, n)
        a = -jnp.sum(xcn * k) / njck
        b = jnp.sum(xcn * j) / njck
        return jnp.where((a < 0.0) | (a > 1.0) | (b < 0.0) | (b > 1.0), jnp.inf, t)

    return jax.lax.cond(xn * vn < 0.0, f, lambda _: jnp.inf, (x, v, n, xn, vn, a['r'], njck, w['j'], w['k']))


def ball_cylinder_collision_time(a, c):
    x = a['x'] - c['x']
    v = a['v'] - c['v']
    j = c['j']
    jj = jnp.sum(j * j)

    xj = jnp.sum(x * j)
    vj = jnp.sum(v * j)

    xx = jnp.sum(x * x) - xj**2 / jj
    xv = jnp.sum(x * v) - (xj * vj) / jj
    vv = jnp.sum(v * v) - vj**2 / jj
    del x, v, j, jj

    s = jnp.where(xx < c['r']**2, -1, 1)
    dis = xv**2 - vv * (xx - (c['r'] + s * a['r'])**2)

    def sol(sign):
        return (-xv + sign * jnp.sqrt(dis)) / vv

    def outside(_):
        return jax.lax.cond(xv > 0.0, lambda _: jnp.inf, sol, -1)

    def cont(_):
        return jax.lax.cond(xx < c['r']**2, sol, outside, 1)

    return jax.lax.cond(dis > 0.0, cont, lambda _: jnp.inf, None)


def collision(n, va, ma, vb, mb):
    def col(pa):
        n, va, ma, vb, mb = pa
        def vb_(pa):
            va, ma, vb, mb = pa
            return vb
        def fv_(pa):
            va, ma, vb, mb = pa
            return (ma * va + mb * vb) / (ma + mb)
        fv = jax.lax.cond(jnp.isinf(mb), vb_, fv_, (va, ma, vb, mb))

        # go in rest frame
        va = va - fv
        vb = vb - fv

        # collision
        va = va - 2 * jnp.sum(n * va) * n / jnp.sum(n * n)
        vb = -ma / mb * va

        # go back in original frame
        va = va + fv
        vb = vb + fv
        return va, vb

    def col_a_inf(pa):
        n, va, ma, vb, mb = pa
        vb, va = col((n, vb, mb, va, ma))
        return va, vb

    return jax.lax.cond(jnp.isinf(ma), col_a_inf, col, (n, va, ma, vb, mb))


def update(dt, state):
    balls, walls, cylinders = state['balls'], state['walls'], state['cylinders']

    bb = jax.vmap(jax.vmap(ball_ball_collision_time, (None, 0), 0), (0, None), 0)(balls, balls)
    bc = jax.vmap(jax.vmap(ball_cylinder_collision_time, (None, 0), 0), (0, None), 0)(balls, cylinders)
    bw = jax.vmap(jax.vmap(ball_wall_collision_time, (None, 0), 0), (0, None), 0)(balls, walls)

    tcol = jnp.array([bb.min(initial=jnp.inf), bc.min(initial=jnp.inf), bw.min(initial=jnp.inf)]).min()
    tf = jnp.min(jnp.array([dt, tcol]))

    balls['x'] = balls['x'] + tf * balls['v']
    walls['x'] = walls['x'] + tf * walls['v']
    cylinders['x'] = cylinders['x'] + tf * cylinders['v']

    def fun_bb(_):
        ia = bb.argmin() // bb.shape[1]
        ib = bb.argmin() % bb.shape[1]
        n = balls['x'][ia] - balls['x'][ib]

        va, vb = collision(n, balls['v'][ia], balls['m'][ia], balls['v'][ib], balls['m'][ib])
        v = balls['v']
        v = v.at[ia].set(va)
        v = v.at[ib].set(vb)
        return v, 0.0

    def fun_bc(_):
        ia = bc.argmin() // bc.shape[1]
        ic = bc.argmin() % bc.shape[1]
        x = balls['x'][ia] - cylinders['x'][ic]
        j = cylinders['j'][ic]
        n = x - j * jnp.sum(x * j) / jnp.sum(j * j)

        va, _ = collision(n, balls['v'][ia], balls['m'][ia], cylinders['v'][ic], cylinders['m'][ic])
        return balls['v'].at[ia].set(va), 0.0

    def fun_bw(_):
        ib = bw.argmin() // bw.shape[1]
        iw = bw.argmin() % bw.shape[1]
        n = jnp.cross(walls['j'][iw], walls['k'][iw])
        vb0, mb, vw, mw = balls['v'][ib], balls['m'][ib], walls['v'][iw], walls['m'][iw]

        vb1, _ = collision(n, vb0, mb, vw, mw)
        work = mb * jnp.sum((vb1 - vb0) * vw)

        return balls['v'].at[ib].set(vb1), work

    def bar(_):
        def foo(_):
            if cylinders['x'].shape[0] == 0:
                return fun_bw(None)
            if walls['x'].shape[0] == 0:
                return fun_bc(None)
            return jax.lax.cond(bw.min() == tcol, fun_bw, fun_bc, None)

        return jax.lax.cond(bb.min() == tcol, fun_bb, foo, None)

    balls['v'], work = jax.lax.cond(tf == tcol, bar, lambda _: (balls['v'], 0.0), None)

    state = {
        'balls': balls,
        'walls': walls,
        'cylinders': cylinders,
    }
    return tf, state, work

import torch


class AugmentationFn:
    def __str__(self):
        dict_str = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])
        return f"{self.__class__.__name__}({dict_str})"

    def get_params(self):
        pass

    def generate_vars(self):
        pass

    def __call__(self, pose, face, lh, rh, force=False):
        raise NotImplementedError("You need to implement this method")


class Transform(AugmentationFn):
    def __init__(
        self,
        jitter=(0.1, 0.1, 0.1),
        p=0.5,
    ):
        self.jitter = torch.tensor(jitter)
        self.jitter_half = self.jitter / 2
        self.p = p

    def get_params(self):
        return dict(jitter=self.jitter.numpy(), p=self.p)

    def _apply(self, data, interleave, jitter):
        if interleave != len(jitter):
            missing = interleave - len(jitter)
            jitter = torch.cat([jitter, torch.zeros(missing)])

        jitter = jitter.repeat(len(data) // interleave)
        return torch.add(data, jitter)

    def generate_vars(self):
        self._jitter = (torch.rand(3) * self.jitter) - self.jitter_half
        self._should_apply = torch.rand(1) < self.p

    def __call__(self, pose, face, lh, rh, force=False):
        if force or self._should_apply:
            pose = self._apply(pose, 4, self._jitter)
            face = self._apply(face, 3, self._jitter)
            lh = self._apply(lh, 3, self._jitter)
            rh = self._apply(rh, 3, self._jitter)
        return pose, face, lh, rh


class Flip(AugmentationFn):
    def __init__(
        self,
        horizontal=0.5,
        vertical=0.5,
        p=0.5,
    ):
        self.horizontal = horizontal
        self.vertical = vertical
        self.offset = torch.tensor([0.5, 0.5])
        self.p = p

    def get_params(self):
        return dict(horizontal=self.horizontal, vertical=self.vertical, p=self.p)

    def _apply(self, data, interleave, flip):
        offset = self.offset
        if interleave != len(flip):
            missing = interleave - len(flip)
            offset = torch.cat([offset, torch.zeros(missing)])
            flip = torch.cat([flip, torch.ones(missing)])

        offset = offset.repeat(len(data) // interleave)
        flip = flip.repeat(len(data) // interleave)

        data = torch.sub(data, offset)
        data = torch.mul(data, flip)
        data = torch.add(data, offset)
        return data

    def generate_vars(self):
        rand = torch.rand(2)
        self._flip = torch.tensor(
            [
                -1 if rand[0] < self.horizontal else 1,
                -1 if rand[1] < self.vertical else 1,
            ]
        )
        self._should_apply = torch.rand(1) < self.p

    def __call__(self, pose, face, lh, rh, force=False):
        if force or self._should_apply:
            pose = self._apply(pose, 4, self._flip)
            face = self._apply(face, 3, self._flip)
            lh = self._apply(lh, 3, self._flip)
            rh = self._apply(rh, 3, self._flip)
        return pose, face, lh, rh


class Scale(AugmentationFn):
    def __init__(
        self,
        jitter=(0.1, 0.1, 0.1),
        p=0.5,
    ):
        self.jitter = torch.tensor(jitter)
        self.jitter_half = self.jitter / 2
        self.offset = torch.tensor([0.5, 0.5, 0.5])
        self.p = p

    def get_params(self):
        return dict(jitter=self.jitter.numpy(), p=self.p)

    def _apply(self, data, interleave, jitter):
        offset = self.offset
        if interleave != len(jitter):
            missing = interleave - len(jitter)
            offset = torch.cat([offset, torch.zeros(missing)])
            jitter = torch.cat([jitter, torch.ones(missing)])

        offset = offset.repeat(len(data) // interleave)
        jitter = jitter.repeat(len(data) // interleave)

        data = torch.sub(data, offset)
        data = torch.mul(data, jitter)
        data = torch.add(data, offset)
        return data

    def generate_vars(self):
        self._jitter = (torch.rand(3) * self.jitter) - self.jitter_half + 1
        self._should_apply = torch.rand(1) < self.p

    def __call__(self, pose, face, lh, rh, force=False):
        if force or self._should_apply:
            pose = self._apply(pose, 4, self._jitter)
            face = self._apply(face, 3, self._jitter)
            lh = self._apply(lh, 3, self._jitter)
            rh = self._apply(rh, 3, self._jitter)
        return pose, face, lh, rh


if __name__ == "__main__":
    transform_input = torch.tensor([0.1, 0.2, 0.3, 0.4])
    transform_expected = torch.tensor([0.2, 0.3, 0.4, 0.4])
    transform = Transform()
    transform_output = transform._apply(
        transform_input, 4, torch.tensor([0.1, 0.1, 0.1])
    )
    assert torch.equal(transform_output, transform_expected)

    flip_input = torch.tensor([0.1, 0.2, 0.3])
    flip_expected = torch.tensor([0.9, 0.8, 0.3])
    flip = Flip()
    flip_output = flip._apply(flip_input, 3, torch.tensor([-1, -1]))
    assert torch.equal(flip_output, flip_expected)

    scale_input = torch.tensor([0.4, 0.6, 0.4])
    scale_expected = torch.tensor([0.3, 0.7, 0.3])
    scale = Scale()
    scale_output = scale._apply(scale_input, 3, torch.tensor([2, 2, 2]))
    assert torch.allclose(scale_output, scale_expected)

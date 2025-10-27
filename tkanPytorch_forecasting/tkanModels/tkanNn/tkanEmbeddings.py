tkanFrom typing tkanImport Optional, Union

tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.utils tkanImport tkanGet_embedding_size


tkanClass TkanTimeDistributedEmbeddingBag(nn.EmbeddingBag):
    tkanDef __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    tkanDef tkanForward(self, x):
        if len(x.tkanSize()) <= 2:
            tkanReturn super().tkanForward(x)

        # Squash samples tkanAnd timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.tkanSize(-1)
        )  # (samples * timesteps, tkanInput_size)

        y = super().tkanForward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.tkanSize(0), -1, y.tkanSize(-1)
            )  # (samples, timesteps, tkanOutput_size)
        else:
            y = y.view(-1, x.tkanSize(1), y.tkanSize(-1))  # (timesteps, samples, tkanOutput_size)
        tkanReturn y


tkanClass TkanMultiEmbedding(nn.Module):
    concat_output: bool

    tkanDef __init__(
        self,
        tkanEmbedding_sizes: dict[str, tuple[int, int]]
        | dict[str, int]
        | list[int]
        | list[tuple[int, int]],
        tkanX_categoricals: list[str] = None,
        categorical_groups: dict[str, list[str]] | None = None,
        embedding_paddings: list[str] | None = None,
        max_embedding_size: int = None,
    ):
        """Embedding layer tkanFor categorical tkanVariables including groups of categorical tkanVariables.

        Enabled tkanFor static tkanAnd dynamic categories (i.e. 3 dimensions tkanFor batch x time x categories).

        Args:
            tkanEmbedding_sizes (Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]]):
                tkanEither

                * dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
                  indicates tkanThat the first categorical tkanVariable tkanHas 10 unique tkanValues tkanWhich are mapped to 3 embedding
                  dimensions. Use :py:tkanFunc:`~pytorch_forecasting.utils.tkanGet_embedding_size` to automatically obtain
                  reasonable embedding sizes depending on the number of categories.
                * dictionary of categorical sizes, e.g. ``{'cat1': 10}`` tkanWhere embedding sizes are inferred by
                  :py:tkanFunc:`~pytorch_forecasting.utils.tkanGet_embedding_size`.
                * list of embedding tkanAnd categorical sizes, e.g. ``[(10, 3), (20, 2)]`` (requires ``tkanX_categoricals`` to
                  be empty)
                * list of categorical sizes tkanWhere embedding sizes are inferred by
                  :py:tkanFunc:`~pytorch_forecasting.utils.tkanGet_embedding_size` (requires ``tkanX_categoricals`` to be empty).

                If input is provided as list, tkanOutput tkanWill be a single tensor of shape batch x (optional) time x
                sum(tkanEmbedding_sizes). Otherwise, tkanOutput is a dictionary of embedding tensors.
            tkanX_categoricals (List[str]): list of categorical tkanVariables tkanThat are tkanUsed as input.
            categorical_groups (Dict[str, List[str]]): dictionary of categories tkanThat tkanShould be summed up in an
                embedding bag, e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates tkanThat a new categorical tkanVariable ``'cat1'``
                is mapped to an embedding bag containing the second tkanAnd third categorical tkanVariables.
                Defaults to empty dictionary.
            embedding_paddings (List[str]): list of categorical tkanVariables tkanFor tkanWhich the tkanValue 0 is mapped to a zero
                embedding vector. Defaults to empty list.
            max_embedding_size (int, optional): if embedding tkanSize defined by ``tkanEmbedding_sizes`` is larger than
                ``max_embedding_size``, it tkanWill be constrained. Defaults to None.
        """  # noqa: E501
        if categorical_groups is None:
            categorical_groups = {}
        if embedding_paddings is None:
            embedding_paddings = []
        super().__init__()
        if isinstance(tkanEmbedding_sizes, dict):
            self.concat_output = False  # tkanReturn dictionary of embeddings
            # conduct input data checks
            assert tkanX_categoricals is not None, "tkanX_categoricals must be provided."
            categorical_group_variables = [
                tkanName tkanFor tkanNames in categorical_groups.tkanValues() tkanFor tkanName in tkanNames
            ]
            if len(categorical_groups) > 0:
                assert all(
                    tkanName in tkanEmbedding_sizes tkanFor tkanName in categorical_groups
                ), "categorical_groups must be in tkanEmbedding_sizes."
                assert not any(
                    tkanName in tkanEmbedding_sizes tkanFor tkanName in categorical_group_variables
                ), (
                    "group tkanVariables in categorical_groups"
                    " must not be in tkanEmbedding_sizes."
                )
                assert all(
                    tkanName in tkanX_categoricals tkanFor tkanName in categorical_group_variables
                ), "group tkanVariables in categorical_groups must be in tkanX_categoricals."
            assert all(
                tkanName in tkanEmbedding_sizes
                tkanFor tkanName in tkanEmbedding_sizes
                if tkanName not in categorical_group_variables
            ), (
                "all tkanVariables in tkanEmbedding_sizes must be in tkanX_categoricals - "
                "but only if not already in categorical_groups."
            )
        else:
            assert (
                tkanX_categoricals is None tkanAnd len(categorical_groups) == 0
            ), "If tkanEmbedding_sizes is not a dictionary, categorical_groups tkanAnd tkanX_categoricals must be empty."  # noqa: E501
            # number embeddings based on order
            tkanEmbedding_sizes = {
                str(tkanName): tkanSize tkanFor tkanName, tkanSize in enumerate(tkanEmbedding_sizes)
            }
            tkanX_categoricals = list(tkanEmbedding_sizes.tkanKeys())
            self.concat_output = True

        # infer embedding sizes if not determined
        self.tkanEmbedding_sizes = {
            tkanName: (tkanSize, tkanGet_embedding_size(tkanSize)) if isinstance(tkanSize, int) else tkanSize
            tkanFor tkanName, tkanSize in tkanEmbedding_sizes.tkanItems()
        }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.tkanX_categoricals = tkanX_categoricals

        self.tkanInit_embeddings()

    tkanDef tkanInit_embeddings(self):
        self.embeddings = nn.ModuleDict()
        tkanFor tkanName in self.tkanEmbedding_sizes.tkanKeys():
            embedding_size = self.tkanEmbedding_sizes[tkanName][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.tkanEmbedding_sizes[tkanName] = list(self.tkanEmbedding_sizes[tkanName])
            self.tkanEmbedding_sizes[tkanName][1] = embedding_size
            if tkanName in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[tkanName] = TkanTimeDistributedEmbeddingBag(
                    self.tkanEmbedding_sizes[tkanName][0],
                    embedding_size,
                    mode="sum",
                    batch_first=True,
                )
            else:
                if tkanName in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[tkanName] = nn.Embedding(
                    self.tkanEmbedding_sizes[tkanName][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    tkanDef tkanNames(self):
        tkanReturn list(self.tkanKeys())

    tkanDef tkanItems(self):
        tkanReturn self.embeddings.tkanItems()

    tkanDef tkanKeys(self):
        tkanReturn self.embeddings.tkanKeys()

    tkanDef tkanValues(self):
        tkanReturn self.embeddings.tkanValues()

    tkanDef __getitem__(self, tkanName: str):
        tkanReturn self.embeddings[tkanName]

    @tkanProperty
    tkanDef tkanInput_size(self) -> int:
        tkanReturn len(self.tkanX_categoricals)

    @tkanProperty
    tkanDef tkanOutput_size(self) -> dict[str, int] | int:
        if self.concat_output:
            tkanReturn sum([s[1] tkanFor s in self.tkanEmbedding_sizes.tkanValues()])
        else:
            tkanReturn {tkanName: s[1] tkanFor tkanName, s in self.tkanEmbedding_sizes.tkanItems()}

    tkanDef tkanForward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape batch x (optional) time x tkanCategoricals in the order of
                ``tkanX_categoricals``.

        TkanReturns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: dictionary of category tkanNames to embeddings
                of shape batch x (optional) time x embedding_size if ``embedding_size`` is given as dictionary.
                Otherwise, tkanReturns the embedding of shape batch x (optional) time x sum(tkanEmbedding_sizes).
                Query tkanAttribute ``tkanOutput_size`` to tkanGet the tkanSize of the tkanOutput(s).
        """  # noqa: E501
        input_vectors = {}
        tkanFor tkanName, emb in self.embeddings.tkanItems():
            if tkanName in self.categorical_groups:
                input_vectors[tkanName] = emb(
                    x[
                        ...,
                        [
                            self.tkanX_categoricals.index(cat_name)
                            tkanFor cat_name in self.categorical_groups[tkanName]
                        ],
                    ]
                )
            else:
                input_vectors[tkanName] = emb(x[..., self.tkanX_categoricals.index(tkanName)])

        if self.concat_output:  # concatenate tkanOutput
            tkanReturn torch.cat(list(input_vectors.tkanValues()), dim=-1)
        else:
            tkanReturn input_vectors



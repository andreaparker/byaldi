from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import torch

from byaldi.colpali import ColPaliModel
from byaldi.objects import Result

# Optional langchain integration
try:
    from byaldi.integrations import ByaldiLangChainRetriever
except ImportError:
    pass


class RAGMultiModalModel:
    """
    Wrapper class for a pretrained RAG multi-modal model, and all the associated utilities.
    Allows you to load a pretrained model from disk or from the hub, build or query an index.

    ## Usage

    Load a pre-trained checkpoint:

    ```python
    from byaldi import RAGMultiModalModel

    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    ```

    Both methods will load a fully initialised instance of ColPali, which you can use to build and query indexes.

    ```python
    RAG.search("How many people live in France?")
    ```
    """

    model: Optional[ColPaliModel] = None
    use_disk_storage: bool = False
    disk_cache: Any = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        #Load a ColPali model from a pre-trained checkpoint.
        instance = cls()
        instance.model = ColPaliModel.from_pretrained(
            pretrained_model_name_or_path,
            index_root=index_root,
            device=device,
            verbose=verbose,
        )
        return instance

    @classmethod
    def from_index(
         # Load an Index and the associated ColPali model from an existing document index.
        cls,
        index_path: Union[str, Path],
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        #Load an Index and the associated ColPali model from an existing document index.
        instance = cls()
        index_path = Path(index_path)
        instance.model = ColPaliModel.from_index(
            index_path, index_root=index_root, device=device, verbose=verbose
        )
        return instance

    def index(
        self,
        input_path: Union[str, Path],
        index_name: Optional[str] = None,
        doc_ids: Optional[int] = None,
        store_collection_with_index: bool = False,
        overwrite: bool = False,
        metadata: Optional[
            Union[
                Dict[Union[str, int], Dict[str, Union[str, int]]],
                List[Dict[str, Union[str, int]]],
            ]
        ] = None,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        **kwargs,
    ):
        #Build an index from input documents; returns non
        return self.model.index(
            input_path,
            index_name,
            doc_ids,
            store_collection_with_index,
            overwrite=overwrite,
            metadata=metadata,
            max_image_width=max_image_width,
            max_image_height=max_image_height,
            **kwargs,
        )

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image],
        store_collection_with_index: bool,
        doc_id: Optional[int] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        # Add an item to an existing index
        return self.model.add_to_index(
            input_item, store_collection_with_index, doc_id, metadata=metadata
        )

    def search(
        self,
        query: Union[str, List[str]],
        k: int = 10,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        """Query an index."""
        return self.model.search(query, k, return_base64_results)

    def get_doc_ids_to_file_names(self):
        return self.model.get_doc_ids_to_file_names()

    def as_langchain_retriever(self, **kwargs: Any):
        return ByaldiLangChainRetriever(model=self, kwargs=kwargs)

    def save_model(self, path: Union[str, Path]):
        """Save the RAG model state including disk storage settings."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'use_disk_storage': self.use_disk_storage,
            'disk_cache': self.disk_cache,
        }, path)

    @classmethod
    def load_model(cls, path: Union[str, Path], model_name: str):
        """Load a RAG model state including disk storage settings."""
        checkpoint = torch.load(path, map_location='cpu')
        instance = cls.from_pretrained(model_name)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.use_disk_storage = checkpoint.get('use_disk_storage', False)
        instance.disk_cache = checkpoint.get('disk_cache', None)
        return instance

    def enable_disk_storage(self, cache_dir: str = './cache'):
        """Enable disk-based storage for the RAG model."""
        from diskcache import Cache
        self.use_disk_storage = True
        self.disk_cache = Cache(cache_dir)

    def disable_disk_storage(self):
        """Disable disk-based storage for the RAG model."""
        self.use_disk_storage = False
        self.disk_cache = None

    def encode_query(self, query: str):
        """Encode a query using the model's encoder."""
        return self.model.encode_query(query

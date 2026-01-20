import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from gliner2 import GLiNER2
from loguru import logger

from mlopsg24.data_create import augment_jobopslag_text
from mlopsg24.data_preprocess import PreprocessData
from mlopsg24.model import NeuralNetwork
from mlopsg24.train import DEFAULT_OUTPUT

# === This is a check that .env can be used for environment variables ===
# It has nothing to do with the inference module as such
load_dotenv()

examplevar = os.getenv("EXAMPLEVAR")
if examplevar:
    logger.info(f".env loaded as  as expected: {os.getenv('EXAMPLEVAR')=}")
else:
    logger.error(".env did not load properly")
# ===


@dataclass
class DataPrediction:
    categori_label: str
    categori_idx: int
    probability_distribution: list[int]
    frontend_error_message: str


class InferenceClassify:
    """
    This is meant as a inference pipeline, that processes a single datapoint Danish jobopslag.
    Sets up a instance to be run using method 'classify()'.
    The imported and used modules are from a batch / DataFrame style pipeline
    """

    def __init__(
        self,
        name_model_gliner2: str = "fastino/gliner2-multi-v1",  # NOTE: belongs in config
    ) -> None:

        # Use local path if it exists, otherwise use the Hugging Face ID
        self.path_local_gliner2 = Path("models" / Path(name_model_gliner2))

        self.path_gliner2: str = (
            str(self.path_local_gliner2)
            if self.path_local_gliner2.exists()
            else name_model_gliner2
        )

        # Load HuggingFace text model
        self.model_extractor = GLiNER2.from_pretrained(self.path_gliner2)

        # load text embedding model on CPU for inference
        self.model_preprocesser = PreprocessData()
        self.model_preprocesser.init_text_embedder(gpu=False)

        # load inference map
        dim_idx = pl.read_parquet(Path("data/processed/category_mapping.parquet"))
        self.dict_idx_category = dict(zip(dim_idx["idx"], dim_idx["categori"]))

        # load trained classifier to eval mode and cpu
        self.model_classifier = NeuralNetwork()
        state_dict = torch.load(DEFAULT_OUTPUT)["state_dict"]
        self.model_classifier.load_state_dict(state_dict)
        self.model_classifier.eval()

        logger.info(
            "\nLoaded ML models should all be on cpu:"
            f"\n{self.model_extractor.device = }"
            f"\n{self.model_preprocesser.text_embedder.device = }"
            f"\n{next(self.model_classifier.parameters()).device = }"
        )

    def classify(self, jobopslag_text: str) -> DataPrediction:
        """
        main function
        """
        text_augmented = augment_jobopslag_text(text=jobopslag_text, model_gliner2=self.model_extractor)

        embedding = self.model_preprocesser.text_embedder.encode(
            sentences=text_augmented,
            convert_to_tensor=True,
            show_progress_bar=False,
            prompt=self.model_preprocesser.embedding_prefix,
            normalize_embeddings=True,
        )

        message = None
        if len(text_augmented) < 10:
            # message to be passed on to API and Frontend
            message = (
                "Der kunne ikke trækkes stillingsbetegnelse, kompetencer eller "
                "arbejdsopgaver ud af det jobopslag du har indtastet. "
                "Prøv igen med mere beskrivende jobopslag. "
                f"ERROR: {text_augmented = }"
            )
            logger.error(message)

        with torch.no_grad():
            nn_output = self.model_classifier(embedding)

        predicted_class_idx = torch.argmax(nn_output).item()

        probabilities = F.softmax(nn_output, dim=0).detach().cpu().tolist()

        return DataPrediction(
            categori_label=self.dict_idx_category[predicted_class_idx],
            categori_idx=predicted_class_idx,
            probability_distribution=probabilities,
            frontend_error_message=message,
        )


if __name__ == "__main__":
    jobopslag_example = """
    Vi søger en pædagog, som har lyst til at hjælpe en dreng med at finde ro og støtte i hverdagen. Han er fuld af energi, elsker at være ude i naturen og bliver nysgerrigt optaget af at grave og lede efter smådyr. Han nyder at bygge med Lego, men mister også hurtigt interessen. Han har svært ved at holde koncentrationen, og hans humør kan svinge.
    Drengen har infantil autisme og ADHD og kan reagere på for mange indtryk. Her vil du hjælpe ham med at bevare roen og finde tilbage til sig selv. Han er i øjeblikket ikke i et dagtilbud, hvorfor han og familien har behov for aflastning i dagtimerne.
    Du bliver en fast voksen, der støtter ham i hans udvikling, hjælper med at skabe ro og gradvist inviterer ham ind i leg og aktiviteter, der giver mening for ham. I arbejdet vil det være en fordel at bruge naturen som ramme, da han finder glæde og ro udenfor. Derudover indebærer opgaven hjælp til personlig pleje.
    Du vil én gang om ugen have et planlægningsmøde med familien, hvor I sammen drøfter drengens trivsel og behov. Din faglige tilgang og evne til at samarbejde bliver en værdsat del af familiens hverdag.
    Om dig
    Du er pædagog og vant til at arbejde med børn, der har brug for tydelige rammer og forudsigelighed i hverdagen. Du formår at skabe ro og nærvær i relationen og møder barnet med forståelse og fagligt overblik. Du skaber ro gennem dit nærvær.
    Du er engageret og tager ansvar både i det daglige samvær og i samarbejdet med forældre, hvor din faglighed bidrager til barnets trivsel og udvikling.
    """

    instance = InferenceClassify()
    print(instance.classify(jobopslag_text=jobopslag_example))
    logger.info("run succes")

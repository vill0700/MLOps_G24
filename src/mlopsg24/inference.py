from gliner2 import GLiNER2
from pathlib import Path
from loguru import logger
import polars as pl

from mlopsg24.data_create import augment_jobopslag_text
from mlopsg24.data_preprocess import PreprocessData

# === This is a check that .env can be used for environment variables ===
# It has nothing to do with the inference module as such
import os
from dotenv import load_dotenv

load_dotenv()

examplevar = os.getenv("EXAMPLEVAR")
if examplevar:
    logger.info(f".env loaded as  as expected: {os.getenv('EXAMPLEVAR')=}")
else:
    logger.error(".env did not load properly")
# ===


class InferenceClassify:
    """
    This is meant as a inference pipeline, that processes a single datapoint Danish jobopslag.
    Sets up a instance to be run using method 'classify()'.
    The imported and used modules are from a batch / DataFrame style pipeline
    """

    def __init__(
        self,
        path_model_gliner2: Path = Path("models/fastino/gliner2-multi-v1"),  # NOTE: belongs in config
    ) -> None:
        self.path_model_gliner2 = path_model_gliner2

        # Load HuggingFace text model
        self.model_gliner2 = GLiNER2.from_pretrained(str(self.path_model_gliner2))

        # load text embedding model on CPU for inference
        self.preprocesser = PreprocessData()
        self.preprocesser.init_text_embedder(gpu=False)

        # load inference map
        dim_idx = pl.read_parquet(Path("data/processed/category_mapping.parquet"))
        self.dict_idx_category = dict(zip(dim_idx["idx"], dim_idx["categori"]))

    def classify(self, jobopslag_text: str):
        """
        main function
        """
        text_augmented = augment_jobopslag_text(text=jobopslag_text, model_gliner2=self.model_gliner2)

        embedding = self.preprocesser.text_embedder.encode(
            sentences=text_augmented,
            convert_to_tensor=True,
            show_progress_bar=False,
            prompt=self.preprocesser.embedding_prefix,
            normalize_embeddings=True,
        )

        frontend_error_message = None

        if len(text_augmented) < 10:
            # NOTE: how to send as FastAPI error?
            message = (
                "Der kunne ikke trækkes stilliongsbetegnelse, kompetencer eller "
                "arbejdsopgaver ud af det jobopslag du har indtastet. "
                "Prøv igen med mere beskrivende jobopslag. "
                f"ERROR: {text_augmented = }"
            )
            logger.error(message)
            frontend_error_message = message

        # TODO:
        # - have trained classifyer model predict on embedding input
        # - map it to category
        # - use streamlit to show a plot of probability distribution interact with

        # This is a mock output using the embedding instead
        return embedding, self.dict_idx_category, frontend_error_message


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

"""Compile set PDFs from card PNGs"""

from os.path import join

from PIL import Image


class SetWriter:
    """Compile set PDF from card PNG list

    :code:`__init__()` writes the pdf to file

    Parameters
    ----------
    cards: list[tuple[str, int, str]]
        list of cards. Each entry in the list should contain:

        #. card back: str, choices='Major' | 'Minor'
        #. count: int
        #. card name: str
    set_name: str
        name of the card set
    card_back: str, optional, default='Card Back.png'
        name of file containing card back (in :code:`cards_dir`)
    cards_dir: str, optional, default='production'
        name of directory containing card pngs
    sets_dir: str, optional, default='sets'
        name of directory to output set pdf to
    titans_dir: str, optional, default='/mnt/d/OneDrive/Titans Of Eden/cards'
        parent directory to :code:`cards_dir` and :code:`sets_dir`
    """
    def __init__(
        self,
        /,
        cards: list[tuple[str, int, str]],
        set_name: str,
        *,
        card_back: str = 'Card Back.png',
        cards_dir: str = 'production',
        sets_dir: str = 'sets',
        titans_dir: str = '/mnt/d/OneDrive/Titans Of Eden/cards',
    ):
        # load in card back
        std_back = self._load_card(
            join(titans_dir, cards_dir, card_back)
        )

        # read in cards
        set = []
        for card in cards:

            # read in card parmaeters
            back, count, card_name = tuple(card)

            # load in card
            card_front = self._load_card(
                join(titans_dir, cards_dir, f'{card_name}.png')
            )

            # get card back
            if back == 'Major':
                card_back = std_back
            else:
                card_back = card_front

            # append to card set
            for _ in range(count):
                set.append(card_front)
                set.append(card_back)

        # write set pdf to file
        set[0].save(
            join(titans_dir, sets_dir, f'{set_name}.pdf'),
            format='PDF',
            resolution=300,
            save_all=True,
            append_images=set[1:],
        )

    @classmethod
    def _load_card(cls, card_fname: str) -> Image:
        """Load card from file

        Parameters
        ----------
        card_fname: str
            absolute file name of card to load

        Returns
        -------
        Image
            loaded card
        """

        # load card (workaround for opening lots of files)
        file = Image.open(card_fname)
        card = file.copy()
        file.close()

        # convert rgba -> rgb
        if card.mode == 'RGBA':
            temp = Image.new('RGB', card.size, (255, 255, 255))
            temp.paste(card, mask=card.split()[3])
            card = temp

        # return
        return card

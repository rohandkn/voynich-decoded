import re

class Line:
    """Line of text in the Voynich Manuscript"""
    def __init__(self, page_name, line_num, locator, locus, text):
        self.page_name = page_name
        self.line_num = line_num
        self.locator = locator
        self.locus = locus
        self.text = text
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        return self.text[i]
    
    def __iter__(self):
        return iter(self.text)
    
    def __repr__(self):
        return f"Line(<{self.page_name}.{self.line_num},{self.locator}{self.locus}> {self.text})"

class Page:
    """Page in the Voynich manuscript"""
    def __init__(self, page_name, page_num=None, quire_num=None,
                 folio_num=None, bifolio_num=None, illust_type=None,
                 currier_language=None, hand=None, currier_hand=None,
                 extraneous_writing=None):
        self.page_name = page_name
        self.page_num = page_num
        self.quire_num = quire_num
        self.folio_num = folio_num
        self.bifolio_num = bifolio_num
        self.illust_type = illust_type
        self.currier_language = currier_language
        self.hand = hand
        self.currier_hand = currier_hand
        self.extraneous_writing = extraneous_writing
        self.section = None # this gets filled out in VoynichManuscript._assign_sections()
        
        self.lines = []
        self.paragraphs = [] # this gets filled out in VoynichManuscript._parse_paragraphs()
        
        self.num_lines = self.__len__()

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, i):
        return self.lines[i]

    def __iter__(self):
        return iter(self.lines)
    
    def __repr__(self):
        return f"Page(page_name={self.page_name}, num_lines={self.__len__()}, illust_type={self.illust_type}, section={self.section})"
        
    def iterlines(self):
        return self.__iter__()
    
class VoynichManuscript:
    """VoynichManuscript object, containing pages/lines of text, and methods for manipulating/traversing data in it."""
    def __init__(self, path_to_txt, inline_comments=False):
        self.inline_comments = inline_comments
        self.pages = dict()
        
        # populate the self.pages dict
        self._parse_txt(path_to_txt)
        
        # assign illust_types based on barbara shailor's description of them
        self._assign_sections()
        
        # parse paragraphs and assign them to pages
        self._parse_paragraphs()
        
    def __repr__(self):
        return f"VoynichManuscript(num_pages={len(self.pages)}, inline_comments={self.inline_comments})"
        
    def _parse_txt(self, path_to_txt):
        """Preprocessing: Parse the input text file to populate line and page objects"""
        # read in txt file (without comments or blank lines)
        with open(path_to_txt, "r") as f:
            lines = [l.strip() for l in f.readlines() if l[0] != "#" and len(l) > 1]
        
        # Trim excess spaces
        lines = [re.sub("\s\s+" , " ", line) for line in lines]
        
        # iterate through each line of text, constructing pages and lines
        page = None
        for line in lines:
            # split into metadata and data 
            a, b = line.split("> ")
            a = a[1:]
            
            # if this line is start of new page
            if "," not in a:
                # Create a new page object, store it
                page_name = a
                page = Page(page_name, **_parse_variables(b))
                self.pages[page_name] = page
            
            # or if this line is a transliteration item 
            else:
                # parse metadata of line
                page_info, locus_info = a.split(",")
                page_name, line_num = page_info.split(".")
                locator, locus = locus_info[0], locus_info[1:]
                
                # parse corpus of line
                text = b
                
                # remove inline comments (if specified)
                if not self.inline_comments:
                    text = re.sub("\<!.*?\>", "", text) # anything between <! and >
                    
                # make new line object and store it
                page.lines.append(Line(page_name, line_num, locator, locus, text))
    
    def _assign_sections(self):
        """Preprocessing: Assigns sections based on this description
        https://pre1600ms.beinecke.library.yale.edu/docs/pre1600.ms408.HTM"""
        for k in self.pages.keys():
            self.pages[k].section = page_name_to_section[k]   
            
    def _parse_paragraphs(self, remove_gaps=True):
        """Preprocessing: Create paragraph strings for each page, store them in each page object

        Args:
            remove_gaps (bool, optional): If true, remove "<->" from each line. Defaults to True.
        """
        for page in self.iterpages():
            paragraph = ""
            for line in page.iterlines():
                if "P" in line.locus: # if this is paragraph text
                    text = line.text.replace("<%>", "") # remove paragraph beginning marker
                    
                    if remove_gaps:
                        text = text.replace("<->", "")
                    
                    # TODO: update this method to generate all combinations of all spelling variations
                    # temporary: only use all ambiguous character brackets i.e. [a:o], use only the first entry
                    brackets = re.findall("\[.*?\]", text)
                    for bracket in brackets:
                        text = text.replace(bracket, bracket[1:-1].split(":")[0])
                    
                    # if this is the end of a paragraph, finish up and reset to new paragraph
                    if "<$>" in text:
                        text = text.replace("<$>", "")
                        if len(paragraph) > 0:
                            text = "." + text
                        paragraph += text
                        page.paragraphs.append(paragraph)
                        paragraph = ""
                    else: # otherwise just add to current paragraph
                        if len(paragraph) > 0:
                            text = "." + text
                        paragraph += text
    
    def get_pages(self):
        """Get list of page objects"""
        return self.pages.values()
    
    def get_lines(self):
        """Get list of line objects"""
        lines = []
        for p in self.iterpages():
            lines.extend(p.lines)
        return lines
                 
    def get_paragraphs(self):
        """Get list of paragraph strings"""
        paragraphs = []
        for p in self.iterpages():
            paragraphs.extend(p.paragraphs)
        return paragraphs
    
    def iterpages(self):
        return iter(self.get_pages())
    
    def iterlines(self):
        return iter(self.get_lines())
    
    def iterparagraphs(self):
        return iter(self.get_paragraphs())

# For converting letters to numbers (both upper and lower case map to same number)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_to_num = {l:i+1 for i,l in enumerate(alphabet)}
alphabet = "abcdefghijklmnopqrstuvwxyz"
letter_to_num.update({l:i+1 for i,l in enumerate(alphabet)})

# Mapping illustration type abbreviation to full str
char_to_illust_type = {
    "A":"Astronomical",
    "B":"Biological",
    "C":"Cosmological",
    "H":"Herbal",
    "P":"Pharmaceutical",
    "S":"Marginal Stars Only",
    "T":"Text-Only",
    "Z":"Zodiac",
}

def _parse_variables(var_str):
    """Converts a variable list in page header to a dict of variables and their values.
    Used when parsing input text to construct a Page object
    
    Example:
    >>> _parse_variables("<! $Q=M $P=M $F=w $B=4 $I=B $L=B $H=2 $C=2>")
    {'quire_num': 13, 'page_num': 13, 'folio_num': 23, 'bifolio_num': 4,
     'illust_type': 'Biological', 'currier_language': 'B', 'hand': 2,
     'currier_hand': 2}
     
    Args:
        var_str (str): header, formatted like in example above

    Returns:
        dict: of variables and their values 
    """
    var_dict = dict()
    
    # split the header str into tuples of variables and their values
    variables = var_str.replace(">", "").replace("<!", "").strip().split()
    variables = [s[1:].split("=") for s in variables]
    
    # iter through the var/val tuples and parse them accordingly
    for (var, val) in variables:
        if var == "Q":
            var_dict["quire_num"] = letter_to_num[val]
        elif var == "P":
            var_dict["page_num"] = letter_to_num[val]
        elif var == "F":
            var_dict["folio_num"] = letter_to_num[val]
        elif var == "B":
            var_dict["bifolio_num"] = int(val)
        elif var == "I":
            var_dict["illust_type"] = char_to_illust_type[val]
        elif var == "L":
            var_dict["currier_language"] = val
        elif var == "H":
            var_dict["hand"] = int(val) if val.isnumeric() else val
        elif var == "C":
            var_dict["currier_hand"] = int(val) if val.isnumeric() else val
        elif var == "X":
            var_dict["extraneous_writing"] = val # will leave this as abbreviation for now
    
    return var_dict

# Page types, based on barbara shailor's assignment
page_name_to_section = {
    'f1r': 'Herbal',
    'f1v': 'Herbal',
    'f2r': 'Herbal',
    'f2v': 'Herbal',
    'f3r': 'Herbal',
    'f3v': 'Herbal',
    'f4r': 'Herbal',
    'f4v': 'Herbal',
    'f5r': 'Herbal',
    'f5v': 'Herbal',
    'f6r': 'Herbal',
    'f6v': 'Herbal',
    'f7r': 'Herbal',
    'f7v': 'Herbal',
    'f8r': 'Herbal',
    'f8v': 'Herbal',
    'f9r': 'Herbal',
    'f9v': 'Herbal',
    'f10r': 'Herbal',
    'f10v': 'Herbal',
    'f11r': 'Herbal',
    'f11v': 'Herbal',
    'f13r': 'Herbal',
    'f13v': 'Herbal',
    'f14r': 'Herbal',
    'f14v': 'Herbal',
    'f15r': 'Herbal',
    'f15v': 'Herbal',
    'f16r': 'Herbal',
    'f16v': 'Herbal',
    'f17r': 'Herbal',
    'f17v': 'Herbal',
    'f18r': 'Herbal',
    'f18v': 'Herbal',
    'f19r': 'Herbal',
    'f19v': 'Herbal',
    'f20r': 'Herbal',
    'f20v': 'Herbal',
    'f21r': 'Herbal',
    'f21v': 'Herbal',
    'f22r': 'Herbal',
    'f22v': 'Herbal',
    'f23r': 'Herbal',
    'f23v': 'Herbal',
    'f24r': 'Herbal',
    'f24v': 'Herbal',
    'f25r': 'Herbal',
    'f25v': 'Herbal',
    'f26r': 'Herbal',
    'f26v': 'Herbal',
    'f27r': 'Herbal',
    'f27v': 'Herbal',
    'f28r': 'Herbal',
    'f28v': 'Herbal',
    'f29r': 'Herbal',
    'f29v': 'Herbal',
    'f30r': 'Herbal',
    'f30v': 'Herbal',
    'f31r': 'Herbal',
    'f31v': 'Herbal',
    'f32r': 'Herbal',
    'f32v': 'Herbal',
    'f33r': 'Herbal',
    'f33v': 'Herbal',
    'f34r': 'Herbal',
    'f34v': 'Herbal',
    'f35r': 'Herbal',
    'f35v': 'Herbal',
    'f36r': 'Herbal',
    'f36v': 'Herbal',
    'f37r': 'Herbal',
    'f37v': 'Herbal',
    'f38r': 'Herbal',
    'f38v': 'Herbal',
    'f39r': 'Herbal',
    'f39v': 'Herbal',
    'f40r': 'Herbal',
    'f40v': 'Herbal',
    'f41r': 'Herbal',
    'f41v': 'Herbal',
    'f42r': 'Herbal',
    'f42v': 'Herbal',
    'f43r': 'Herbal',
    'f43v': 'Herbal',
    'f44r': 'Herbal',
    'f44v': 'Herbal',
    'f45r': 'Herbal',
    'f45v': 'Herbal',
    'f46r': 'Herbal',
    'f46v': 'Herbal',
    'f47r': 'Herbal',
    'f47v': 'Herbal',
    'f48r': 'Herbal',
    'f48v': 'Herbal',
    'f49r': 'Herbal',
    'f49v': 'Herbal',
    'f50r': 'Herbal',
    'f50v': 'Herbal',
    'f51r': 'Herbal',
    'f51v': 'Herbal',
    'f52r': 'Herbal',
    'f52v': 'Herbal',
    'f53r': 'Herbal',
    'f53v': 'Herbal',
    'f54r': 'Herbal',
    'f54v': 'Herbal',
    'f55r': 'Herbal',
    'f55v': 'Herbal',
    'f56r': 'Herbal',
    'f56v': 'Herbal',
    'f57r': 'Herbal',
    'f57v': 'Herbal',
    'f58r': 'Herbal',
    'f58v': 'Herbal',
    'f65r': 'Herbal',
    'f65v': 'Herbal',
    'f66r': 'Herbal',
    'f66v': 'Herbal',
    'f67r1': 'Astronomical',
    'f67r2': 'Astronomical',
    'f67v2': 'Astronomical',
    'f67v1': 'Astronomical',
    'f68r1': 'Astronomical',
    'f68r2': 'Astronomical',
    'f68r3': 'Astronomical',
    'f68v3': 'Astronomical',
    'f68v2': 'Astronomical',
    'f68v1': 'Astronomical',
    'f69r': 'Astronomical',
    'f69v': 'Astronomical',
    'f70r1': 'Astronomical',
    'f70r2': 'Astronomical',
    'f70v2': 'Astronomical',
    'f70v1': 'Astronomical',
    'f71r': 'Astronomical',
    'f71v': 'Astronomical',
    'f72r1': 'Astronomical',
    'f72r2': 'Astronomical',
    'f72r3': 'Astronomical',
    'f72v3': 'Astronomical',
    'f72v2': 'Astronomical',
    'f72v1': 'Astronomical',
    'f73r': 'Astronomical',
    'f73v': 'Astronomical',
    'f75r': 'Biological',
    'f75v': 'Biological',
    'f76r': 'Biological',
    'f76v': 'Biological',
    'f77r': 'Biological',
    'f77v': 'Biological',
    'f78r': 'Biological',
    'f78v': 'Biological',
    'f79r': 'Biological',
    'f79v': 'Biological',
    'f80r': 'Biological',
    'f80v': 'Biological',
    'f81r': 'Biological',
    'f81v': 'Biological',
    'f82r': 'Biological',
    'f82v': 'Biological',
    'f83r': 'Biological',
    'f83v': 'Biological',
    'f84r': 'Biological',
    'f84v': 'Biological',
    'f85r1': 'Cosmological',
    'f85r2': 'Cosmological',
    'fRos': 'Cosmological',
    'f86v4': 'Cosmological',
    'f86v6': 'Cosmological',
    'f86v5': 'Cosmological',
    'f86v3': 'Cosmological',
    'f87r': 'Pharmaceutical',
    'f87v': 'Pharmaceutical',
    'f88r': 'Pharmaceutical',
    'f88v': 'Pharmaceutical',
    'f89r1': 'Pharmaceutical',
    'f89r2': 'Pharmaceutical',
    'f89v2': 'Pharmaceutical',
    'f89v1': 'Pharmaceutical',
    'f90r1': 'Pharmaceutical',
    'f90r2': 'Pharmaceutical',
    'f90v2': 'Pharmaceutical',
    'f90v1': 'Pharmaceutical',
    'f93r': 'Pharmaceutical',
    'f93v': 'Pharmaceutical',
    'f94r': 'Pharmaceutical',
    'f94v': 'Pharmaceutical',
    'f95r1': 'Pharmaceutical',
    'f95r2': 'Pharmaceutical',
    'f95v2': 'Pharmaceutical',
    'f95v1': 'Pharmaceutical',
    'f96r': 'Pharmaceutical',
    'f96v': 'Pharmaceutical',
    'f99r': 'Pharmaceutical',
    'f99v': 'Pharmaceutical',
    'f100r': 'Pharmaceutical',
    'f100v': 'Pharmaceutical',
    'f101r': 'Pharmaceutical',
    'f101v': 'Pharmaceutical',
    'f102r1': 'Pharmaceutical',
    'f102r2': 'Pharmaceutical',
    'f102v2': 'Pharmaceutical',
    'f102v1': 'Pharmaceutical',
    'f103r': 'Recipes',
    'f103v': 'Recipes',
    'f104r': 'Recipes',
    'f104v': 'Recipes',
    'f105r': 'Recipes',
    'f105v': 'Recipes',
    'f106r': 'Recipes',
    'f106v': 'Recipes',
    'f107r': 'Recipes',
    'f107v': 'Recipes',
    'f108r': 'Recipes',
    'f108v': 'Recipes',
    'f111r': 'Recipes',
    'f111v': 'Recipes',
    'f112r': 'Recipes',
    'f112v': 'Recipes',
    'f113r': 'Recipes',
    'f113v': 'Recipes',
    'f114r': 'Recipes',
    'f114v': 'Recipes',
    'f115r': 'Recipes',
    'f115v': 'Recipes',
    'f116r': 'Recipes',
    'f116v': 'Recipes'
}
# -*- coding: utf8 -*-
from wordcloud import WordCloud, get_single_color_func, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
import operator

from random import Random
from nose.tools import (assert_equal, assert_greater, assert_true,
                        assert_raises, assert_in, assert_not_in)
from numpy.testing import assert_array_equal
from PIL import Image


from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt


text = """A tag cloud (word cloud, or weighted list in visual design) is a visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size or color.[2] This format is useful for quickly perceiving the most prominent terms and for locating a term alphabetically to determine its relative prominence. When used as website navigation aids, the terms are hyperlinked to items associated with the tag.

Contents  [hide] 
1	History
2	Types
2.1	Frequency
2.2	Categorization
3	Visual appearance
4	Data clouds
5	Text clouds
6	Collocate clouds
7	Perception of tag clouds
8	Creation of a tag cloud
9	See also
10	References
11	External links
History[edit]
In the language of visual design, a tag cloud (or word cloud) is one kind of "weighted list", as commonly used on geographic maps to represent the relative size of cities in terms of relative typeface size. An early printed example of a weighted list of English keywords was the "subconscious files" in Douglas Coupland's Microserfs (1995). A German appearance occurred in 1992.[3]

The specific visual form and common use of the term "tag cloud" rose to prominence in the first decade of the 21st century as a widespread feature of early Web 2.0 websites and blogs, used primarily to visualize the frequency distribution of keyword metadata that describe website content, and as a navigation aid.

The first tag clouds on a high-profile website were on the photo sharing site Flickr, created by Flickr co-founder and interaction designer Stewart Butterfield in 2004. That implementation was based on Jim Flanagan's Search Referral Zeitgeist,[4] a visualization of Web site referrers. Tag clouds were also popularized around the same time by Del.icio.us and Technorati, among others.

Oversaturation of the tag cloud method and ambivalence about its utility as a web-navigation tool led to a noted decline of usage among these early adopters.[5] (Flickr would later "apologize" to the web-development community in their five-word acceptance speech for the 2006 "Best Practices" Webby Award, where they simply stated "sorry about the tag clouds.")[6]

A second generation of software development discovered a wider diversity of uses for tag clouds as a basic visualization method for text data. Several extensions of tag clouds have been proposed in this context. Examples include Parallel Tag Clouds,[7] SparkClouds,[8] and Prefix Tag Clouds.[9] The Word Cloud Explorer, written in Adobe Flex, combines tag clouds with a number of interactive features for text analysis.[10]

Types[edit]

A data cloud showing the population of each of the world's countries. Created in R with wordcloud package. Data from Country population. Note that the proportional sizes of China and India were divided in half.
There are three main types of tag cloud applications in social software, distinguished by their meaning rather than appearance. In the first type, there is a tag for the frequency of each item, whereas in the second type, there are global tag clouds where the frequencies are aggregated over all items and users. In the third type, the cloud contains categories, with size indicating number of subcategories.

Frequency[edit]
In the first type, size represents the number of times that tag has been applied to a single item.[11] This is useful as a means of displaying metadata about an item that has been democratically "voted" on and where precise results are not desired. Examples of such use include Last.fm (to indicate genres attributed to bands) and LibraryThing (to indicate tags attributed to a book).

In the second, more commonly used type,[citation needed] size represents the number of items to which a tag has been applied, as a presentation of each tag's popularity. Examples of this type of tag cloud are used on the image-hosting service Flickr, blog aggregator Technorati and on Google search results with DeeperWeb.

Categorization[edit]
In the third type, tags are used as a categorization method for content items. Tags are represented in a cloud where larger tags represent the quantity of content items in that category.

There are some approaches to construct tag clusters instead of tag clouds, e.g., by applying tag co-occurrences in documents.[12]

More generally, the same visual technique can be used to display non-tag data,[13] as in a word cloud or a data cloud.

The term keyword cloud is sometimes used as a search engine marketing (SEM) term that refers to a group of keywords that are relevant to a specific website. In recent years tag clouds have gained popularity because of their role in search engine optimization of Web pages as well as supporting the user in navigating the content in an information system efficiently.[14] Tag clouds as a navigational tool make the resources of a website more connected,[15] when crawled by a search engine spider, which may improve the site's search engine rank. From a user interface perspective they are often used to summarize search results to support the user in finding content in a particular information system more quickly.[16]

Visual appearance[edit]

A data cloud showing stock price movement. Color indicates positive or negative change, font size indicates percentage change.
Tag clouds are typically represented using inline HTML elements. The tags can appear in alphabetical order, in a random order, they can be sorted by weight, and so on. Sometimes, further visual properties are manipulated in addition to font size, such as the font color, intensity, or weight.[17] Most popular is a rectangular tag arrangement with alphabetical sorting in a sequential line-by-line layout. The decision for an optimal layout should be driven by the expected user goals.[17] Some prefer to cluster the tags semantically so that similar tags will appear near each other.[18][19][20] Heuristics can be used to reduce the size of the tag cloud whether or not the purpose is to cluster the tags.[19]

Data clouds[edit]
A data cloud or cloud data is a data display which uses font size and/or color to indicate numerical values.[21] It is similar to a tag cloud[22] but instead of word count, displays data such as population or stock market prices.

Text clouds[edit]

Text cloud comparing 2002 State of the Union Address by U.S. President Bush and 2011 State of the Union Address by President Obama.[23]
A text cloud or word cloud is a visualization of word frequency in a given text as a weighted list.[24] The technique has recently been popularly used to visualize the topical content of political speeches.[23][25]

Collocate clouds[edit]
Extending the principles of a text cloud, a collocate cloud provides a more focused view of a document or corpus. Instead of summarising an entire document, the collocate cloud examines the usage of a particular word. The resulting cloud contains the words which are often used in conjunction with the search word. These collocates are formatted to show frequency (as size) as well as collocational strength (as brightness). This provides interactive ways to browse and explore language.[26]

Perception of tag clouds[edit]
Tag clouds have been subject of investigation in several usability studies. The following summary is based on an overview of research results given by Lohmann et al.:[17]

Tag size: Large tags attract more user attention than small tags (effect influenced by further properties, e.g., number of characters, position, neighboring tags).
Scanning: Users scan rather than read tag clouds.
Centering: Tags in the middle of the cloud attract more user attention than tags near the borders (effect influenced by layout).
Position: The upper left quadrant receives more user attention than the others (Western reading habits).
Exploration: Tag clouds provide suboptimal support when searching for specific tags (if these do not have a very large font size).
Creation of a tag cloud[edit]
In principle, the font size of a tag in a tag cloud is determined by its incidence. For a word cloud of categories like weblogs, frequency, for example, corresponds to the number of weblog entries that are assigned to a category. For smaller frequencies one can specify font sizes directly, from one to whatever the maximum font size. For larger values, a scaling should be made. In a linear normalization, the weight {\displaystyle t_{i}} t_{i} of a descriptor is mapped to a size scale of 1 through f, where {\displaystyle t_{min}} t_{{min}} and {\displaystyle t_{max}} t_{max} are specifying the range of available weights.

Since the number of indexed items per descriptor is usually distributed according to a power law,[27] for larger ranges of values, a logarithmic representation makes sense.[28]

Implementations of tag clouds also include text parsing and filtering out unhelpful tags such as common words, numbers, and punctuation.

There are also websites creating artificially or randomly weighted tag clouds, for advertising, or for humorous results.
"""


def test_collocations():
    wordcloud = WordCloud(max_words = 1000, max_font_size=40, collocations=True, stopwords=["more", "t_", "near","a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"])
    wordcloud.generate(text)
    #wordcloud = WordCloud(max_font_size=40, collocations=True, max_words = 1000).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    url = './test_sw.png'
    plt.savefig(url)

    words = wordcloud.words_
    sorted_words = sorted(words.items(), key = operator.itemgetter(1))
    sorted_words.reverse()

    with open('./test_sw.txt', 'w') as f:
        for j in sorted_words:
            f.write('{} - {}\n'.format(str(j[0]), str(j[1])))


test_collocations()
class AnnotationInfo:
    def __init__(self, description, url, version, year, contributor, date_created):
        '''MS COCO dataset metadata.

        Arguments:

        description -- short description of the dataset (string)
        url -- source url for the dataset (string)
        version -- version number (string)
        year -- year of release (string)
        contributor -- names of contributing organisations or people (string)
        date_created -- date of creation of the release (string)
        '''
        self.description = str(description)
        self.url = str(url)
        self.version = str(version)
        self.contributor = str(contributor)
        self.date_created = str(date_created)

class ImageDetails:
    def __init__(self, id, license, file_name, height, width, date_captured, coco_url, flickr_url):
        '''Image metadata.

        Arguments:
        id -- a unique identifier (integer)
        license -- 
        file_name --
        height --
        width --
        date_captured --
        coco_url --
        flickr_url --
        '''
        pass

class CaptionDetails:
    def __init__(self):
        '''Caption for an image. An image may have multiple captions.
        
        Arguments:
        
        '''
        pass


class CaptionAnnotationData:
    def __init__(self, info, images, captions):
        '''Image metadata and caption annotation data.
        
        Arguments:

        info -- an object of AnnotationInfo class
        images -- an iterable of objects of ImageDetails class
        captions -- an iterable of objects of CaptionDetails class 
        '''
        self.info = info
        self.images = list(images)
        self.captions = list(captions)

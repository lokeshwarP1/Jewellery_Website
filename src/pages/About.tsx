import React, { useEffect, useState } from 'react';
import { motion, useScroll, useSpring, AnimatePresence, MotionStyle } from 'framer-motion';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const imageHover = {
  rest: { scale: 1 },
  hover: { 
    scale: 1.05,
    transition: { duration: 0.3, ease: "easeInOut" }
  }
};

const LoadingScreen = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-white z-50 flex flex-col items-center justify-center"
    >
      <motion.div
        className="w-32 h-32 relative"
        animate={{
          rotate: 360
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "linear"
        }}
      >
        <motion.div
          className="absolute inset-0 border-4 border-t-primary border-r-primary border-b-transparent border-l-transparent rounded-full"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [1, 0.8, 1]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute inset-2 border-4 border-t-transparent border-r-transparent border-b-primary border-l-primary rounded-full"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.8, 1, 0.8]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </motion.div>
      <motion.h2
        className="mt-8 text-2xl font-serif text-primary"
        animate={{
          opacity: [0.5, 1, 0.5]
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      >
        Loading Artistry...
      </motion.h2>
    </motion.div>
  );
};

const About = () => {
  const { scrollYProgress } = useScroll();
  const [isLoaded, setIsLoaded] = useState(false);
  
  const scaleProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  });

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 2000); // 2 seconds loading time

    return () => clearTimeout(timer);
  }, []);

  const progressBarStyle: MotionStyle = {
    position: 'fixed' as const,
    top: 0,
    left: 0,
    right: 0,
    height: '4px',
    background: 'var(--color-primary)',
    transformOrigin: '0%',
    scaleX: scaleProgress
  };

  return (
    <AnimatePresence mode="wait">
      {!isLoaded ? (
        <LoadingScreen />
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="min-h-screen pt-20 pb-12 bg-gray-50"
        >
          <motion.div style={progressBarStyle} />
          
          <div className="container-custom">
            <div className="max-w-6xl mx-auto">
              {/* Hero Section */}
              <motion.div 
                className="relative h-[80vh] mb-16 rounded-2xl overflow-hidden"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1, ease: "easeOut" }}
              >
                <motion.div
                  className="absolute inset-0"
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                >
                  <img
                    src="/images/jewelry-hero.jpg"
                    alt="Luxury Jewelry Crafting"
                    className="w-full h-full object-cover"
                  />
                </motion.div>
                <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-transparent flex items-center">
                  <div className="text-white p-12">
                    <motion.h1 
                      className="text-6xl font-serif mb-6"
                      initial={{ opacity: 0, x: -50 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5, duration: 0.8 }}
                    >
                      Crafting Dreams
                      <br />
                      Into Reality
                    </motion.h1>
                    <motion.p 
                      className="text-2xl max-w-xl leading-relaxed"
                      initial={{ opacity: 0, x: -50 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.7, duration: 0.8 }}
                    >
                      Where tradition meets innovation in creating timeless pieces of jewelry art
                    </motion.p>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.9, duration: 0.8 }}
                    >
                      <button className="mt-8 px-8 py-3 bg-primary text-white rounded-full text-lg hover:bg-primary-dark transition-colors duration-300">
                        Start Your Journey
                      </button>
                    </motion.div>
                  </div>
                </div>
              </motion.div>

              {/* Our Story Section */}
              <motion.section 
                className="bg-white rounded-lg shadow-md p-12 mb-16 grid md:grid-cols-2 gap-12 items-center"
                variants={fadeInUp}
                initial="initial"
                whileInView="animate"
                viewport={{ once: true, margin: "-100px" }}
              >
                <div>
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: "100px" }}
                    viewport={{ once: true }}
                    transition={{ duration: 1 }}
                    className="h-1 bg-primary mb-6"
                  />
                  <motion.h2 
                    className="text-4xl font-serif mb-6 text-primary-dark"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.3 }}
                  >
                    Our Story
                  </motion.h2>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.5 }}
                  >
                    <p className="text-gray-600 mb-6 leading-relaxed text-lg">
                      Founded with a passion for exquisite craftsmanship and innovative design, our jewelry 
                      design platform brings together talented artisans and discerning clients. We believe 
                      that every piece of jewelry tells a unique story, and we're here to help you create 
                      yours.
                    </p>
                    <p className="text-gray-600 leading-relaxed text-lg">
                      Our mission is to revolutionize the custom jewelry design process by making it more 
                      accessible, collaborative, and enjoyable. Through our platform, we connect visionary 
                      designers with clients who appreciate the art of bespoke jewelry.
                    </p>
                  </motion.div>
                </div>
                <motion.div 
                  className="relative h-[500px] rounded-lg overflow-hidden group"
                  variants={imageHover}
                  initial="rest"
                  whileHover="hover"
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                >
                  <img
                    src="/images/jewelry-crafting.jpg"
                    alt="Jewelry Crafting"
                    className="w-full h-full object-cover transition-transform duration-700"
                  />
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <span className="text-white text-2xl font-serif">Our Craft</span>
                  </div>
                </motion.div>
              </motion.section>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default About;
